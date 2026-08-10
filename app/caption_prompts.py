"""Dependency-free Gemini prompt policy for Day0 video dataset captioning."""

from __future__ import annotations


VIDEO_ACTION_CAPTION_REQUIREMENTS = (
    "Write one concise, factual sentence that describes the actual visible action and how it "
    "progresses through the clip. Identify the visibly adult participant count and roles when "
    "clear; describe the pose, body orientation, important spatial or contact relationships, and "
    "the meaningful change in position or motion. State POV/first-person perspective or the actual "
    "camera angle and framing. Mention expression, outfit, hairstyle, accessories, setting, and "
    "background only when visible and useful. Keep permanent identity description minimal and do "
    "not describe face shape or body type. Use direct literal language rather than euphemisms, "
    "inferred intent, isolated anatomy tags, quality tags, or negative phrasing. Vary the action "
    "verb, clause order, and sentence structure across clips instead of repeating one fixed action "
    "phrase, while preserving the exact visible meaning; necessary participant and anatomical nouns "
    "may repeat when accuracy requires them."
)


DEFAULT_VIDEO_CAPTION_INSTRUCTION = (
    "You are a world-class AI model specialist preparing short-video captions for an H3 "
    "action/concept LoRA dataset. Watch the complete clip before captioning it. "
    + VIDEO_ACTION_CAPTION_REQUIREMENTS
    + " Output only the single-sentence caption with no preamble, labels, H3 prompt sections, "
    "timestamps, or line breaks."
)


SMART_CLIP_INSTRUCTION_TEMPLATE = (
    "You are curating a video LoRA training dataset. Watch this clip and identify up to 3 short "
    "segments where clear physical motion/action happens (not static shots, pans, or dead time). "
    "Each segment should be a single continuous moment, roughly {clip_seconds} seconds long, that "
    "on its own teaches the motion well. For each segment, write a separate dataset caption. "
    + VIDEO_ACTION_CAPTION_REQUIREMENTS
    + " Captions returned for multiple segments must use distinct natural wording while describing "
    "each segment's actual action accurately.{variation_context} Respond with ONLY a JSON array, "
    "no markdown fences, no other text, in this exact shape: "
    "[{{\"start\": <seconds float>, \"end\": <seconds float>, \"caption\": \"<text>\"}}, ...]. If "
    "there isn't enough distinct motion for multiple segments, return fewer entries. Times must fall "
    "within the clip's actual duration."
)


def caption_variation_context(previous_captions: list[str], limit: int = 8) -> str:
    """Give Gemini enough recent context to vary phrasing without sacrificing literal accuracy."""
    cleaned = [" ".join(str(caption).split())[:500] for caption in previous_captions if str(caption).strip()]
    if not cleaned:
        return ""
    recent = cleaned[-limit:]
    examples = "\n".join(f"- {caption}" for caption in recent)
    return (
        "\n\nRecent captions from this same dataset are listed below. Do not copy their complete action "
        "phrases or sentence structure. Use a different accurate formulation for this clip, but "
        "never substitute a less precise action merely to sound different:\n" + examples
    )
