"""WanDirector — Wan2.2/2.1 timeline node with Prompt Relay (v2 clean rewrite).

A friendly single node that does for Wan what LTX Director does for LTX: a visual
timeline drives multi-prompt temporal Prompt Relay over one coherent generation,
with the first (and optional last) timeline image used as the I2V / FLF keyframe.

Design (Redmine #92):
- The Prompt Relay machinery (prompt_relay.py) and attention patching
  (patches.py, already Wan-aware) are reused verbatim — only the latent geometry
  and I2V conditioning are Wan-specific, and those delegate to ComfyUI's native
  Wan nodes via wan_processing.py.
- Phase 2+3 are merged: Wan's start/end-frame conditioning is a single native
  call, so there is no separate "guide" node — the Director outputs sample-ready
  positive / negative / latent.
- MoE: the same temporal mask is patched into BOTH the high- and low-noise models.
- Audio is intentionally absent (Wan has no audio track).

Timeline widget names mirror LTX Director so the shared js/ltx_director.js editor
can drive this node (wired up in Phase 4).
"""
import json
import logging

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)
from .patches import detect_model_type, apply_patches
from .wan_processing import (
    resolve_wan_dims,
    encode_wan_i2v,
    segment_latent_lengths,
)
from .ltx_director import _load_image_tensor  # generic image loader (reused)

log = logging.getLogger(__name__)


def _extract_image_segments(timeline_data, duration_frames):
    """Image segments from the timeline JSON, sorted by start (first = start frame,
    last = end frame for FLF). Excludes segments starting beyond the duration."""
    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
    except (ValueError, TypeError):
        return []
    segs = [
        s for s in tdata.get("segments", [])
        if s.get("type", "image") == "image"
        and (s.get("imageFile") or s.get("imageB64"))
        and int(s.get("start", 0)) < duration_frames
    ]
    segs.sort(key=lambda s: s.get("start", 0))
    return segs


def _empty_conditioning(clip):
    return clip.encode_from_tokens_scheduled(clip.tokenize(""))


class WanDirector(io.ComfyNode):
    """Wan timeline editor with Prompt Relay. Place an image at the start of the
    timeline for I2V (and one at the end for first-last-frame); prompt segments
    drive the temporal attention relay across the single ~5s generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDirector",
            display_name="Wan Director",
            category="WhatDreamsCost",
            description=(
                "Wan2.2/2.1 timeline editor with Prompt Relay. A start-image segment makes "
                "it I2V (start + end = first-last-frame); prompt segments drive temporal "
                "attention masking across one generation. Outputs feed the MoE two-stage "
                "KSamplerAdvanced pair (high → low). Wan has no audio."
            ),
            inputs=[
                io.Model.Input("model_high", tooltip="Wan diffusion model. For MoE 14B this is the high-noise model."),
                io.Model.Input("model_low", optional=True,
                               tooltip="Optional — Wan2.2 MoE 14B low-noise model. Patched with the same relay mask."),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True,
                             tooltip="Required for I2V/FLF (encodes the start/end image) and to size the latent."),
                io.ClipVision.Input("clip_vision", optional=True,
                                    tooltip="Strongly recommended for I2V/FLF: a CLIP-Vision model (e.g. clip_vision_h). "
                                            "The Director CLIP-Vision-encodes its own timeline start/end image, which is "
                                            "the semantic anchor that keeps the output faithful to the input image's "
                                            "features/colors. Without it, only the latent concat anchors the image (weak; "
                                            "the result drifts)."),
                io.Conditioning.Input("negative", optional=True,
                                      tooltip="Optional negative. If unconnected an empty negative is built from clip."),
                io.ClipVisionOutput.Input("clip_vision_start", optional=True,
                                          tooltip="Optional pre-encoded CLIP-Vision output of the start image "
                                                  "(overrides the internal encode)."),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True,
                                          tooltip="Optional pre-encoded CLIP-Vision output of the end image (FLF only)."),
                # --- required timeline widgets (names mirror LTX Director for the shared JS) ---
                io.String.Input("global_prompt", multiline=True, default="",
                                tooltip="Conditions the entire video; anchors persistent characters / scene."),
                io.Int.Input("width", default=0, min=0, max=8192, step=16,
                             tooltip="0 = auto-detect from the start image (snapped to divisible_by). Set for T2V."),
                io.Int.Input("height", default=0, min=0, max=8192, step=16,
                             tooltip="0 = auto-detect from the start image (snapped to divisible_by). Set for T2V."),
                io.Int.Input("length", default=81, min=1, max=10000, step=4,
                             tooltip="Pixel-space frame count. Wan expects (length - 1) %% 4 == 0 (e.g. 81, 77, 49)."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("duration_frames", default=120, min=1, max=10000, step=1,
                             tooltip="Timeline display length in pixel-space frames (editor scale only)."),
                io.Float.Input("duration_seconds", default=5.0, min=0.1, max=1000.0, step=0.01,
                               tooltip="Total timeline duration in seconds (display only)."),
                io.String.Input("timeline_data", default="",
                                tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand)."),
                io.String.Input("local_prompts", multiline=True, default="",
                                tooltip="Auto-populated from the timeline editor (pipe-separated segment prompts)."),
                io.String.Input("segment_lengths", default="",
                                tooltip="Auto-populated from the timeline editor (pixel-space frame counts)."),
                io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                               tooltip="Prompt Relay penalty decay (paper default 0.001)."),
                # --- optional widgets ---
                io.Combo.Input("i2v_backend", options=["native", "fmlf"], default="native", optional=True,
                               tooltip="I2V conditioning backend. native = WanImageToVideo/FLF (no deps). "
                                       "fmlf = Wan22FMLF WanAdvancedI2V (Phase 6 SVI; falls back to native if absent)."),
                io.Float.Input("frame_rate", default=24, min=1, max=240, step=1, optional=True),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.Int.Input("divisible_by", default=16, min=1, max=256, step=1, optional=True,
                             tooltip="Snap auto-detected dimensions to a multiple of this (Wan2.1: 16, 2.2-5B: 32)."),
                io.String.Input("guide_strength", default="", optional=True,
                                tooltip="Unused for Wan; kept for shared timeline-JS compatibility."),
            ],
            outputs=[
                io.Model.Output(display_name="model_high"),
                io.Model.Output(display_name="model_low",
                                tooltip="Patched low-noise model (mirrors model_high if model_low is unconnected)."),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model_high, clip, global_prompt, width, height, length, batch_size,
                duration_frames, duration_seconds, timeline_data, local_prompts, segment_lengths,
                epsilon=1e-3, i2v_backend="native", frame_rate=24, display_mode="seconds",
                divisible_by=16, guide_strength="",
                model_low=None, vae=None, clip_vision=None, negative=None,
                clip_vision_start=None, clip_vision_end=None) -> io.NodeOutput:

        # --- Validate prompt segments ---
        locals_list = [p.strip() for p in (local_prompts or "").split("|")]
        for p in locals_list:
            if not p:
                raise ValueError("There is a segment on the timeline missing a prompt!")
        if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
            raise ValueError("At least one local prompt is required.")

        # --- Architecture (must be Wan) ---
        arch, patch_size, _ = detect_model_type(model_high)
        if arch != "wan":
            raise ValueError(
                f"WanDirector expects a Wan diffusion model on model_high, got arch='{arch}'. "
                "Use LTXDirector for LTX models."
            )

        # --- Resolve start / end images from timeline image segments ---
        img_segs = _extract_image_segments(timeline_data, duration_frames)
        start_image = _load_image_tensor(img_segs[0]) if len(img_segs) >= 1 else None
        end_image = _load_image_tensor(img_segs[-1]) if len(img_segs) >= 2 else None

        # --- CLIP-Vision encode the timeline images (the Wan I2V semantic anchor) ---
        # The Director already holds the images, so it encodes them itself when given a
        # CLIP-Vision model. Explicit clip_vision_* outputs (if wired) take precedence.
        if clip_vision is not None:
            if clip_vision_start is None and start_image is not None:
                clip_vision_start = clip_vision.encode_image(start_image, crop=True)
            if clip_vision_end is None and end_image is not None:
                clip_vision_end = clip_vision.encode_image(end_image, crop=True)

        # --- Target dimensions (0 = auto-detect from start image) ---
        tgt_w, tgt_h = resolve_wan_dims(start_image, width, height, divisible_by)

        # --- Build full relay prompt conditioning (independent of the latent) ---
        raw_tokenizer = get_raw_tokenizer(clip)
        full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        if negative is None:
            negative = _empty_conditioning(clip)

        # --- Build Wan latent + I2V/FLF/T2V conditioning (delegates to native Wan nodes) ---
        positive, negative, latent, latent_frames = encode_wan_i2v(
            positive, negative, vae, tgt_w, tgt_h, length, batch_size,
            start_image=start_image, end_image=end_image,
            clip_vision_start=clip_vision_start, clip_vision_end=clip_vision_end,
            backend=i2v_backend,
        )

        # --- Prompt Relay temporal mask (from the actual latent geometry) ---
        samples = latent["samples"]
        tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

        parsed_lengths = None
        if segment_lengths and segment_lengths.strip():
            pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
            parsed_lengths = segment_latent_lengths(pixel_lengths, latent_frames)

        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
        mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

        log.info(
            "[WanDirector] %dx%d, %d latent frames, %d tokens/frame, %d segments %s, backend=%s",
            tgt_w, tgt_h, latent_frames, tokens_per_frame, len(locals_list), effective_lengths, i2v_backend,
        )

        # --- Patch the same relay mask into both MoE models ---
        patched_high = model_high.clone()
        apply_patches(patched_high, arch, mask_fn)
        if model_low is not None:
            patched_low = model_low.clone()
            apply_patches(patched_low, arch, mask_fn)
        else:
            patched_low = patched_high

        return io.NodeOutput(patched_high, patched_low, positive, negative, latent, float(frame_rate))
