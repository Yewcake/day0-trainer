"""Masterchef Cooking (experimental) -- per-image stuck-image detection and
adaptive per-image LR throttling during training.

The idea: raw per-step loss mostly reflects which noise level got drawn that
step, not whether the model has actually learned a given image (a high-sigma
draw looks "worse" than a low-sigma draw regardless of image quality). So we
track, per image, an EMA of loss *relative to* a running per-noise-level
baseline computed across the whole dataset -- an image whose relative loss
stays elevated for many epochs without improving is a real signal something's
wrong with it (usually a caption that doesn't match what's visible), not just
noise.

Flagged images get their gradient contribution throttled (not zeroed) so one
bad caption can't keep yanking the shared weights around for the whole run,
and get soft-excluded (skipped, reversibly) if they never recover. This file
has no CUDA/model awareness at all -- it only ever sees scalars (loss, sigma,
epoch, an image identifier) and returns a multiplier or a boolean, so it's
safe to bolt onto any optimizer.

No auto-recaption here (v1): flagged images are surfaced for the user to fix
by hand via the existing caption editor, not rewritten automatically.
"""

from __future__ import annotations

SIGMA_BUCKETS = 8
SIGMA_MIN = 0.02
SIGMA_MAX = 0.98
BUCKET_EMA_ALPHA = 0.05  # slow-moving -- this is the "what's normal here" baseline
IMAGE_EMA_BETA = 0.15  # each image is only seen once per epoch, so a few epochs' worth of smoothing

SUSPECT_RATIO = 1.3
SUSPECT_MIN_EPOCHS = 3
STUCK_MIN_EPOCHS = 6  # total epochs elevated (not additional) before suspect -> stuck
EXCLUDE_MIN_EPOCHS = 10  # total epochs elevated before stuck -> excluded
IMPROVING_DROP = 0.05  # relative drop since status started that counts as "improving"

THROTTLE_MULTIPLIER = {"ok": 1.0, "suspect": 0.7, "stuck": 0.4, "excluded": 0.0}


def _bucket_for_sigma(sigma: float) -> int:
    span = SIGMA_MAX - SIGMA_MIN
    frac = (min(max(sigma, SIGMA_MIN), SIGMA_MAX) - SIGMA_MIN) / span
    return min(SIGMA_BUCKETS - 1, int(frac * SIGMA_BUCKETS))


class MasterchefTracker:
    def __init__(self) -> None:
        self.bucket_ema: list[float | None] = [None] * SIGMA_BUCKETS
        self.per_image: dict[str, dict] = {}

    def _image_state(self, image: str) -> dict:
        state = self.per_image.get(image)
        if state is None:
            state = {
                "ratio_ema": 1.0,
                "status": "ok",
                "epochs_in_status": 0,
                "status_start_ratio": 1.0,
                "last_epoch": -1,
            }
            self.per_image[image] = state
        return state

    def observe(self, image: str, loss: float, sigma: float, epoch: int) -> None:
        bucket = _bucket_for_sigma(sigma)
        baseline = self.bucket_ema[bucket]
        if baseline is None:
            self.bucket_ema[bucket] = loss
            baseline = loss
        else:
            baseline = BUCKET_EMA_ALPHA * loss + (1 - BUCKET_EMA_ALPHA) * baseline
            self.bucket_ema[bucket] = baseline

        ratio = loss / baseline if baseline > 1e-8 else 1.0
        state = self._image_state(image)
        state["ratio_ema"] = IMAGE_EMA_BETA * ratio + (1 - IMAGE_EMA_BETA) * state["ratio_ema"]
        state["last_epoch"] = epoch

    def sweep_epoch(self, epoch: int) -> None:
        """Call once per epoch boundary. Escalates/de-escalates status for every
        image observed so far based on its current ratio_ema trend."""
        for state in self.per_image.values():
            if state["status"] == "excluded":
                continue
            elevated = state["ratio_ema"] > SUSPECT_RATIO
            improving = state["ratio_ema"] < state["status_start_ratio"] * (1 - IMPROVING_DROP)

            if improving:
                if state["status"] != "ok":
                    state["status"] = "ok"
                    state["epochs_in_status"] = 0
                    state["status_start_ratio"] = state["ratio_ema"]
                continue

            if not elevated:
                if state["status"] != "ok":
                    state["status"] = "ok"
                    state["epochs_in_status"] = 0
                    state["status_start_ratio"] = state["ratio_ema"]
                continue

            state["epochs_in_status"] += 1
            total_elevated = state["epochs_in_status"]
            if state["status"] == "ok" and total_elevated >= SUSPECT_MIN_EPOCHS:
                state["status"] = "suspect"
            elif state["status"] == "suspect" and total_elevated >= STUCK_MIN_EPOCHS:
                state["status"] = "stuck"
            elif state["status"] == "stuck" and total_elevated >= EXCLUDE_MIN_EPOCHS:
                state["status"] = "excluded"

    def multiplier_for(self, image: str) -> float:
        state = self.per_image.get(image)
        if state is None:
            return 1.0
        return THROTTLE_MULTIPLIER[state["status"]]

    def is_excluded(self, image: str) -> bool:
        state = self.per_image.get(image)
        return bool(state and state["status"] == "excluded")

    def to_status_dict(self) -> dict:
        images = [
            {
                "image": image,
                "status": state["status"],
                "ratio_ema": round(state["ratio_ema"], 3),
                "epochs_in_status": state["epochs_in_status"],
            }
            for image, state in self.per_image.items()
            if state["status"] != "ok"
        ]
        images.sort(key=lambda row: row["ratio_ema"], reverse=True)
        return {"enabled": True, "images": images}
