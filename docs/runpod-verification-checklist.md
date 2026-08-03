RunPod's support reply (2026-08-03) gave a 5-point checklist for getting a template review-ready before requesting official verification. This maps each point to this specific repo/image/template's actual state.

## 1. Template config is "review-ready"

- **Container image**: `ghcr.io/yewcake/day0-trainer:latest`. Confirmed public and pullable anonymously (`docker manifest inspect` succeeds with no auth) — 10.4GB compressed, 21 layers, amd64. No registry credentials needed in the template.
- **Ports**: `8888`, matching the Dockerfile's `EXPOSE 8888` and `start.sh`'s `UI_PORT` default. **Check this in the console, not just the label**: the support message says RunPod's *JupyterLab health check* specifically targets port 8888 — this suggests the template editor may have an actual "Jupyter" port **type/preset** (a real health-check behavior expecting a genuine Jupyter response), not just a cosmetic label. Our container runs a custom FastAPI dashboard on 8888, not real Jupyter (see `docker/start.sh` — it's `uvicorn app.main:app`, no `jupyter` anywhere in the image). If the port is set to a "Jupyter" type in the console, switch it to a generic **HTTP** port type — renaming the label to "Dashboard" (already requested) fixes the display but not a real type mismatch if one exists.
- **Container start command**: leave blank. The image's `Dockerfile` already sets `CMD ["/start.sh"]`, which handles the git pull, env-var checks, and launching uvicorn. Overriding it in the template risks breaking that sequence for no benefit.
- **Disk sizes**: two separate settings, don't conflate them —
  - *Container disk* (holds the pulled image itself): recommend **30GB+**. The image is 10.4GB compressed; CUDA/PyTorch images typically extract to 2-3x their compressed size, so give real headroom.
  - *Volume* (mounted at `/workspace`, holds model cache/datasets/checkpoints): **150GB+**, already documented in the README and template description.

## 2. JupyterLab port 8888

Not applicable in the Jupyter sense — this template doesn't run Jupyter at all, it's a from-scratch FastAPI dashboard that happens to also use port 8888 by convention. See the port-type note above; that's the one real risk this point surfaces.

## 3. Registry accessibility

Already public — confirmed by an anonymous `docker manifest inspect` pull with no credentials, no `unauthorized`/`denied` errors. Nothing to configure.

## 4. Test-deploy before requesting verification

Already done, repeatedly, this session — multiple live pods deployed from this exact template across Krea 2, Ideogram 4, and MiniMax H3 jobs. Container started cleanly each time, the dashboard was reachable on port 8888, and logs were clean apart from real application bugs (all now fixed and pushed — see recent commit history). This is usable evidence to cite when requesting verification.

## 5. Requesting official verification

Platform-side action — has to be done by you via RunPod's support form (Console → the verification request form referenced in their message). Suggested text to paste in:

> Template name: day0-trainer (Day0 Trainer — Krea 2 / Ideogram 4 / MiniMax H3 LoRA Trainer)
> What it does: A self-updating LoRA/LoKr training tool for Krea 2, Ideogram 4, and MiniMax H3 diffusion/video models, with a web dashboard for dataset upload, captioning, training, and checkpoint download. Source: https://github.com/Yewcake/day0-trainer (public).
> Verification checklist already completed: public GHCR image confirmed pullable, port 8888 exposed for the dashboard (not Jupyter — custom FastAPI UI), no start-command override, container disk sized for the image, volume sized for model/dataset storage, and the template has been test-deployed successfully multiple times.

I don't have RunPod console/API access, so I can't submit this form or change the port type/disk size myself — everything above is either already true in the repo/image, or needs your action in the console.
