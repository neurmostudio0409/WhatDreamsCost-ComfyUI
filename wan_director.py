"""WanDirector — Wan2.2 family timeline editor with Prompt Relay support.

Supports:
- Wan2.2 T2V-A14B / I2V-A14B (MoE 14B; two model inputs)
- Wan2.2 FLF (first-last-frame, 14B)
- Wan2.2 TI2V-5B (single 5B model, different VAE)
- Wan2.1 T2V / I2V (backward compatible)

The timeline editor JS is shared with LTX Director — image segments at the
edges of the timeline supply start_image / end_image, prompt segments drive
the Prompt Relay temporal attention masking. Audio segments are ignored.
"""

import logging
import json

import torch

import comfy.model_management
import comfy.latent_formats
import comfy.utils
import node_helpers

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, detect_wan_geometry, apply_patches
from .ltx_director import (
    _load_image_tensor,
    _resize_image,
    _convert_to_latent_lengths,
)

log = logging.getLogger(__name__)


_VARIANTS = ["auto", "t2v-14b", "i2v-14b", "flf-14b", "ti2v-5b"]


def _resolve_variant(variant_input: str, geom: dict, num_image_segments: int) -> str:
    """Resolve 'auto' to a concrete variant based on the model's latent format
    and the number of image segments on the timeline."""
    if variant_input != "auto":
        return variant_input

    if geom["latent_channels"] == 48:
        return "ti2v-5b"

    if geom["sub_type"] == "i2v":
        return "flf-14b" if num_image_segments >= 2 else "i2v-14b"

    if num_image_segments >= 2:
        return "flf-14b"
    if num_image_segments == 1:
        return "i2v-14b"
    return "t2v-14b"


def _snap(val: int, div: int) -> int:
    return max(div, (val // div) * div)


def _resize_and_normalize(tensor, target_w, target_h, divisible_by, resize_method):
    """Resize image tensor to target dims, snapping to divisible_by."""
    src_h, src_w = tensor.shape[1], tensor.shape[2]

    if target_w > 0 and target_h > 0:
        return _resize_image(tensor, target_w, target_h, resize_method, divisible_by)
    if target_w > 0:
        tw = _snap(target_w, divisible_by)
        th = _snap(int(src_h * tw / src_w), divisible_by)
        return _resize_image(tensor, tw, th, "stretch to fit", divisible_by)
    if target_h > 0:
        th = _snap(target_h, divisible_by)
        tw = _snap(int(src_w * th / src_h), divisible_by)
        return _resize_image(tensor, tw, th, "stretch to fit", divisible_by)

    return _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)


def _extract_image_segments(timeline_data_str: str, duration_frames: int):
    """Return list of {tensor, start_frame} sorted by start, dropping any past duration."""
    if not timeline_data_str:
        return []
    try:
        tdata = json.loads(timeline_data_str)
    except Exception:
        return []

    segs = [
        s for s in tdata.get("segments", [])
        if s.get("type", "image") == "image"
        and (s.get("imageFile") or s.get("imageB64"))
        and int(s.get("start", 0)) < duration_frames
    ]
    segs.sort(key=lambda s: int(s.get("start", 0)))
    return segs


def _build_wan_latent(width: int, height: int, length: int, batch_size: int, geom: dict):
    """Generate empty Wan latent of the right shape and downscale."""
    sp = geom["spacial_downscale"]
    ch = geom["latent_channels"]
    latent_t = ((length - 1) // 4) + 1
    samples = torch.zeros(
        [batch_size, ch, latent_t, height // sp, width // sp],
        device=comfy.model_management.intermediate_device(),
    )
    return samples, latent_t


def _encode_image_to_video_size(image_tensor: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Upscale a single image [1, H, W, 3] to (height, width) using ComfyUI utils."""
    return comfy.utils.common_upscale(
        image_tensor.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)


def _apply_i2v_conditioning(positive, negative, vae, width, height, length, start_image, latent_t,
                            clip_vision_output=None):
    """I2V-14B conditioning: concat_latent_image + concat_mask attached to both pos/neg."""
    start_image = _encode_image_to_video_size(start_image, width, height)
    image = torch.ones((length, height, width, start_image.shape[-1]),
                       device=start_image.device, dtype=start_image.dtype) * 0.5
    image[:start_image.shape[0]] = start_image

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = torch.ones(
        (1, 1, latent_t, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=start_image.device, dtype=start_image.dtype,
    )
    mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

    positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
    negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    return positive, negative


def _apply_flf_conditioning(positive, negative, vae, width, height, length, start_image, end_image,
                            latent, clip_vision_start=None, clip_vision_end=None):
    """First-Last-Frame conditioning: encode both ends, mask both, attach to pos/neg."""
    if start_image is not None:
        start_image = _encode_image_to_video_size(start_image[:length], width, height)
    if end_image is not None:
        end_image = _encode_image_to_video_size(end_image[-length:], width, height)

    image = torch.ones((length, height, width, 3)) * 0.5
    mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

    if start_image is not None:
        image[:start_image.shape[0]] = start_image
        mask[:, :, :start_image.shape[0] + 3] = 0.0
    if end_image is not None:
        image[-end_image.shape[0]:] = end_image
        mask[:, :, -end_image.shape[0]:] = 0.0

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

    positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
    negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

    clip_vision_out = clip_vision_start
    if clip_vision_end is not None:
        if clip_vision_out is not None:
            import comfy.clip_vision
            states = torch.cat([clip_vision_out.penultimate_hidden_states,
                                clip_vision_end.penultimate_hidden_states], dim=-2)
            clip_vision_out = comfy.clip_vision.Output()
            clip_vision_out.penultimate_hidden_states = states
        else:
            clip_vision_out = clip_vision_end

    if clip_vision_out is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_out})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_out})

    return positive, negative


def _build_ti2v_5b_latent(vae, width, height, length, batch_size, start_image=None):
    """Wan2.2 TI2V-5B latent: 48 channels, /16 downscale, image is baked into latent with noise_mask."""
    latent_t = ((length - 1) // 4) + 1
    latent = torch.zeros([1, 48, latent_t, height // 16, width // 16],
                         device=comfy.model_management.intermediate_device())

    if start_image is None:
        return {"samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1))}, latent_t

    mask = torch.ones([1, 1, latent_t, latent.shape[-2], latent.shape[-1]],
                      device=comfy.model_management.intermediate_device())
    start_image = _encode_image_to_video_size(start_image[:length], width, height)
    latent_temp = vae.encode(start_image)
    latent[:, :, :latent_temp.shape[-3]] = latent_temp
    mask[:, :, :latent_temp.shape[-3]] *= 0.0

    latent_format = comfy.latent_formats.Wan22()
    latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
    out = {
        "samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1)),
        "noise_mask": mask.repeat((batch_size,) + (1,) * (mask.ndim - 1)),
    }
    return out, latent_t


def _apply_prompt_relay_patch(model, clip, latent_samples, geom, global_prompt, locals_list,
                              parsed_lengths, epsilon, frame_offset=0):
    """Build patched model + positive conditioning from prompt-relay segments.

    frame_offset shifts segment midpoints forward to skip leading reference
    frames in the latent (used by WanAnimateDirector for the trim_latent prefix).

    Returns (patched_model, positive_conditioning).
    """
    arch, patch_size, temporal_stride = detect_model_type(model)

    latent_frames = latent_samples.shape[2]
    generated_frames = max(1, latent_frames - frame_offset)
    tokens_per_frame = (latent_samples.shape[3] // patch_size[1]) * (latent_samples.shape[4] // patch_size[2])

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[WanDirector] Global tokens [0:%d]", token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[WanDirector] Segment %d tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    # Distribute segment lengths over the GENERATED span only (excluding any reference prefix),
    # then shift each segment's midpoint forward by frame_offset via build_segments.
    effective_lengths = distribute_segment_lengths(len(locals_list), generated_frames, parsed_lengths)
    log.info("[WanDirector] Latent %d frames (offset=%d, gen=%d), %d tokens/frame, segments=%s",
             latent_frames, frame_offset, generated_frames, tokens_per_frame, effective_lengths)

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None, frame_offset=frame_offset)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)
    return patched, conditioning


def _empty_cond_from_clip(clip):
    """Build an empty conditioning (used as the default negative)."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(""))


class WanDirector(io.ComfyNode):
    """WYSIWYG timeline for Wan2.2 — same Prompt Relay temporal attention as LTX Director,
    but plumbs the Wan-family conditioning interface (concat_latent_image for I2V/FLF,
    Wan2.2 TI2V-5B latent for 5B). Supports the MoE 14B variant via two model inputs."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDirector",
            display_name="Wan Director",
            category="WhatDreamsCost",
            description=(
                "Wan2.2-family timeline editor. Supports T2V/I2V/FLF-14B (single or MoE) and "
                "TI2V-5B. Place an image segment at the start of the timeline for I2V, and at "
                "both the start and the end for FLF. Prompt segments drive the temporal Prompt "
                "Relay attention masking — same idea as LTX Director."
            ),
            inputs=[
                io.Model.Input("model_high", tooltip="The Wan diffusion model. For MoE 14B this is the high-noise model."),
                io.Model.Input("model_low", optional=True, tooltip="Optional — for Wan2.2 MoE 14B, the low-noise model. Patched with the same prompt-relay mask."),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="Required for I2V / FLF / TI2V-5B (used to encode the start / end images). Also used to decode prev_latent's tail frame when auto-chaining."),
                io.Latent.Input("prev_latent", optional=True,
                                tooltip=(
                                    "Optional. Connect the PREVIOUS chunk's output video latent to auto-chain: "
                                    "the Director decodes its last frame (via the connected vae) and uses it as "
                                    "this chunk's start image, so the new chunk continues seamlessly from where "
                                    "the last one ended. Forces an image-to-video variant (i2v-14b / ti2v-5b). "
                                    "Ignored when the timeline already has a start-image segment."
                                )),
                io.Conditioning.Input("negative", optional=True, tooltip="Optional negative conditioning. If unconnected an empty negative is built from clip."),
                io.ClipVisionOutput.Input("clip_vision_start", optional=True, tooltip="Optional CLIP-Vision output of the start image (Wan I2V / FLF)."),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True, tooltip="Optional CLIP-Vision output of the end image (Wan FLF only)."),
                io.Combo.Input("model_variant", options=_VARIANTS, default="auto",
                               tooltip="'auto' picks from the model's latent format and image-segment count."),
                io.String.Input("global_prompt", multiline=True, default="",
                                tooltip="Conditions the entire video. Anchors persistent characters / scene context."),
                io.Int.Input("width", default=0, min=0, max=8192, step=16,
                             tooltip="Output width. 0 = auto-detect from the start image (I2V/FLF/chained); "
                                     "snapped to `divisible_by`. Set a value to force a size, or for T2V "
                                     "(no start image). Wan2.2-5B usually wants a multiple of 32 (e.g. 1280)."),
                io.Int.Input("height", default=0, min=0, max=8192, step=16,
                             tooltip="Output height. 0 = auto-detect from the start image (I2V/FLF/chained); "
                                     "snapped to `divisible_by`. Set a value to force a size, or for T2V "
                                     "(no start image). Wan2.2-5B usually wants a multiple of 32 (e.g. 704)."),
                io.Int.Input("length", default=81, min=1, max=10000, step=4,
                             tooltip="Pixel-space frame count. Wan models expect (length - 1) %% 4 == 0 (e.g. 81, 77, 49)."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("duration_frames", default=120, min=1, max=10000, step=1,
                             tooltip="Timeline display length in pixel-space frames (visual scale for the editor)."),
                io.Float.Input("duration_seconds", default=5, min=0.1, max=1000.0, step=0.01,
                               tooltip="Total timeline duration in seconds (display only)."),
                io.String.Input("timeline_data", default="",
                                tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand)."),
                io.Boolean.Input("use_custom_audio", default=False, optional=True,
                                 tooltip="Reserved — Wan Director ignores audio segments. Toggle has no effect for Wan."),
                io.String.Input("local_prompts", multiline=True, default="",
                                tooltip="Auto-populated from the timeline editor."),
                io.String.Input("segment_lengths", default="",
                                tooltip="Auto-populated from the timeline editor (pixel-space frame counts)."),
                io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                               tooltip="Penalty decay parameter for Prompt Relay (paper default 0.001)."),
                io.Float.Input("frame_rate", default=24, min=1, max=240, step=1, optional=True),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.String.Input("guide_strength", default="",
                                tooltip="Auto-populated by the editor (unused for Wan; kept for shared JS compat)."),
                io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, optional=True,
                             tooltip="Override target width when processing start/end images. 0 = use `width` from above."),
                io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, optional=True,
                             tooltip="Override target height when processing start/end images. 0 = use `height` from above."),
                io.Combo.Input("resize_method",
                               options=["maintain aspect ratio", "stretch to fit", "pad", "crop"],
                               default="maintain aspect ratio", optional=True),
                io.Int.Input("divisible_by", default=16, min=1, max=256, step=1, optional=True,
                             tooltip="Snap image dimensions to a multiple of this (Wan2.1: 16, Wan2.2-5B: 32)."),
                io.Int.Input("img_compression", default=0, min=0, max=100, step=1, optional=True,
                             tooltip="Reserved (Wan does not need H.264 compression on guide images)."),
            ],
            outputs=[
                io.Model.Output(display_name="model_high"),
                io.Model.Output(display_name="model_low", tooltip="Patched MoE low-noise model (passes through original if model_low is unconnected)."),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model_high, clip, global_prompt, width, height, length, batch_size,
                duration_frames, duration_seconds, timeline_data, local_prompts, segment_lengths,
                epsilon=1e-3, frame_rate=24, display_mode="seconds",
                guide_strength="", custom_width=0, custom_height=0,
                resize_method="maintain aspect ratio", divisible_by=16, img_compression=0,
                model_variant="auto", use_custom_audio=False,
                model_low=None, vae=None, negative=None,
                clip_vision_start=None, clip_vision_end=None, prev_latent=None) -> io.NodeOutput:

        # --- Validate prompt segments ---
        locals_list = [p.strip() for p in local_prompts.split("|")]
        for p in locals_list:
            if not p:
                raise ValueError("There is a segment on the timeline missing a prompt!")
        if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
            raise ValueError("At least one local prompt is required.")

        # --- Geometry detection ---
        arch, _, _ = detect_model_type(model_high)
        if arch != "wan":
            raise ValueError(
                f"WanDirector expects a Wan diffusion model on model_high, got arch='{arch}'. "
                "Use LTXDirector for LTX models."
            )
        geom = detect_wan_geometry(model_high)

        # --- Resolve start_image / end_image from timeline image segments ---
        img_segs = _extract_image_segments(timeline_data, duration_frames)

        # --- Auto-chain: derive a start frame from the previous chunk's tail ---
        # When prev_latent is wired and the timeline has no image segment, decode the previous
        # chunk's LAST latent frame and use it as this chunk's start image, so the chunk continues
        # from where the last one ended. This counts as one image segment, so the variant resolves
        # to an image-to-video mode.
        prev_start = None
        if prev_latent is not None and len(img_segs) == 0:
            if vae is None:
                raise ValueError(
                    "WanDirector: prev_latent is connected but no vae. Connect the Wan VAE so "
                    "the previous chunk's tail frame can be decoded."
                )
            prev_samples = prev_latent["samples"]
            if prev_samples.dim() != 5:
                raise ValueError(
                    f"WanDirector: prev_latent must be a 5D video latent [B,C,T,H,W], got shape "
                    f"{tuple(prev_samples.shape)}."
                )
            decoded = vae.decode(prev_samples[:, :, -1:].contiguous())  # last latent frame only
            if decoded.dim() == 5:  # [B,T,H,W,C] -> drop batch
                decoded = decoded[0]
            prev_start = decoded[-1:]  # single start frame (the continuation point)
            log.info("[WanDirector] Auto-derived start_image from prev_latent tail: %s -> image %s",
                     tuple(prev_samples.shape), tuple(prev_start.shape))

        effective_img_count = len(img_segs) + (1 if prev_start is not None else 0)
        variant = _resolve_variant(model_variant, geom, effective_img_count)

        # Snap width/height to multiples
        # Requested dims. 0 = auto-detect from the start image (snapped to divisible_by).
        # custom_width / custom_height still take precedence over width / height when > 0.
        req_w = custom_width if custom_width > 0 else width
        req_h = custom_height if custom_height > 0 else height

        def _load_seg(seg):
            t = _load_image_tensor(seg)
            # Auto-size: when a dimension is 0, take it from the image's native size.
            nat_h, nat_w = t.shape[1], t.shape[2]
            w = _snap(req_w if req_w > 0 else nat_w, divisible_by)
            h = _snap(req_h if req_h > 0 else nat_h, divisible_by)
            return _resize_and_normalize(t, w, h, divisible_by, resize_method)

        start_image = _load_seg(img_segs[0]) if len(img_segs) >= 1 else None
        end_image = _load_seg(img_segs[-1]) if len(img_segs) >= 2 else None

        # No timeline start image but a previous-chunk tail frame was decoded → use it as the start.
        if start_image is None and prev_start is not None:
            start_image = prev_start

        # Resolve the dims used for latent generation.
        if start_image is not None:
            # Auto: adopt the (resized / decoded) start image's dimensions.
            tgt_h = start_image.shape[1]
            tgt_w = start_image.shape[2]
        else:
            # T2V / no image: can't auto-detect, so explicit width & height are required.
            if req_w <= 0 or req_h <= 0:
                raise ValueError(
                    "WanDirector: no start image to auto-detect dimensions from. Set width and "
                    "height explicitly (T2V), or add a start-image segment to the timeline."
                )
            tgt_w = _snap(req_w, divisible_by)
            tgt_h = _snap(req_h, divisible_by)

        # --- Parsed segment lengths (pixel-space → latent frames) ---
        parsed_lengths = None
        if segment_lengths.strip():
            pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
            # Use Wan's temporal stride of 4 (all variants)
            # We'll compute proper latent-space lengths after we know latent_t.
            parsed_lengths_raw = pixel_lengths
        else:
            parsed_lengths_raw = None

        # --- Build latent (variant-specific) ---
        if variant == "ti2v-5b":
            latent_out, latent_t = _build_ti2v_5b_latent(vae, tgt_w, tgt_h, length, batch_size,
                                                        start_image=start_image)
            samples = latent_out["samples"]
        else:
            samples, latent_t = _build_wan_latent(tgt_w, tgt_h, length, batch_size, geom)
            latent_out = {"samples": samples}

        # Convert pixel-space segment lengths to latent-space frame counts
        if parsed_lengths_raw is not None:
            parsed_lengths = _convert_to_latent_lengths(parsed_lengths_raw, 4, latent_t)

        # --- Apply Prompt Relay patches to model_high and model_low ---
        patched_high, positive = _apply_prompt_relay_patch(
            model_high, clip, samples, geom, global_prompt, locals_list, parsed_lengths, epsilon,
        )

        if model_low is not None:
            patched_low, _ = _apply_prompt_relay_patch(
                model_low, clip, samples, geom, global_prompt, locals_list, parsed_lengths, epsilon,
            )
        else:
            patched_low = model_high

        # --- Negative conditioning (build empty if not provided) ---
        if negative is None:
            negative = _empty_cond_from_clip(clip)

        # --- Variant-specific conditioning attachment ---
        if variant == "i2v-14b":
            if vae is None:
                raise ValueError("WanDirector: i2v-14b needs a VAE connected to encode the start image.")
            if start_image is None:
                raise ValueError("WanDirector: i2v-14b needs at least one image segment on the timeline (start image).")
            positive, negative = _apply_i2v_conditioning(
                positive, negative, vae, tgt_w, tgt_h, length, start_image, latent_t,
                clip_vision_output=clip_vision_start,
            )

        elif variant == "flf-14b":
            if vae is None:
                raise ValueError("WanDirector: flf-14b needs a VAE connected to encode the start/end images.")
            positive, negative = _apply_flf_conditioning(
                positive, negative, vae, tgt_w, tgt_h, length,
                start_image, end_image, samples,
                clip_vision_start=clip_vision_start, clip_vision_end=clip_vision_end,
            )

        elif variant == "ti2v-5b":
            # Conditioning is in the latent (noise_mask); no concat_latent_image attach needed.
            # But clip-vision output can still be attached.
            if clip_vision_start is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_start})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_start})

        # t2v-14b: no extra conditioning attachment

        log.info("[WanDirector] variant=%s, latent_shape=%s, model_low=%s",
                 variant, tuple(samples.shape), "patched" if model_low is not None else "passthrough")

        return io.NodeOutput(patched_high, patched_low, positive, negative, latent_out, float(frame_rate))


# ============================================================================
# Phase 2 — Wan2.2 Advanced Directors (S2V / Animate / VACE)
# ============================================================================
#
# These Directors share the Prompt Relay timeline UI and the underlying Wan
# attention-masking patch, but plumb the variant-specific control conditioning
# (audio embeddings, pose/face videos, control-video VACE context). All three
# use a single model input — they target the corresponding standalone Wan2.2
# checkpoints (not the MoE 14B pair).


def _shared_timeline_inputs():
    """Common Prompt-Relay / timeline-editor widgets reused by every Director.

    Widget names must match what `js/ltx_director.js` reads, since all Director
    nodes share the same Timeline UI extension.
    """
    return [
        io.String.Input("global_prompt", multiline=True, default="",
                        tooltip="Conditions the entire video. Anchors persistent characters / scene context."),
        io.Int.Input("duration_frames", default=120, min=1, max=10000, step=1,
                     tooltip="Timeline display length in pixel-space frames (visual scale only)."),
        io.Float.Input("duration_seconds", default=5, min=0.1, max=1000.0, step=0.01),
        io.String.Input("timeline_data", default="", tooltip="Auto-managed by the editor."),
        io.Boolean.Input("use_custom_audio", default=False, optional=True,
                         tooltip="Reserved — Wan Directors ignore audio segments."),
        io.String.Input("local_prompts", multiline=True, default="", tooltip="Auto-populated from the timeline."),
        io.String.Input("segment_lengths", default="", tooltip="Auto-populated from the timeline."),
        io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001),
        io.Float.Input("frame_rate", default=24, min=1, max=240, step=1, optional=True),
        io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
        io.String.Input("guide_strength", default="", tooltip="Auto-populated (unused for Wan)."),
        io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, optional=True),
        io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, optional=True),
        io.Combo.Input("resize_method",
                       options=["maintain aspect ratio", "stretch to fit", "pad", "crop"],
                       default="maintain aspect ratio", optional=True),
        io.Int.Input("divisible_by", default=16, min=1, max=256, step=1, optional=True),
        io.Int.Input("img_compression", default=0, min=0, max=100, step=1, optional=True),
    ]


def _load_first_image_segment(timeline_data, duration_frames, divisible_by, target_w, target_h,
                              resize_method):
    """Pull the first image segment off the timeline and resize it. Returns None if no segment."""
    img_segs = _extract_image_segments(timeline_data, duration_frames)
    if not img_segs:
        return None
    tensor = _load_image_tensor(img_segs[0])
    return _resize_and_normalize(tensor, target_w, target_h, divisible_by, resize_method)


def _validate_locals(local_prompts: str):
    locals_list = [p.strip() for p in local_prompts.split("|")]
    for p in locals_list:
        if not p:
            raise ValueError("There is a segment on the timeline missing a prompt!")
    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        raise ValueError("At least one local prompt is required.")
    return locals_list


def _common_prompt_relay_setup(model, clip, samples, global_prompt, locals_list,
                                segment_lengths_str, epsilon, frame_offset=0):
    """Run the Prompt-Relay encode + patch pipeline. Returns (patched_model, positive_cond).

    frame_offset (latent frames) skips a reference prefix when distributing segments.
    """
    parsed_lengths = None
    if segment_lengths_str.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths_str.split(",") if x.strip()]
        generated_t = max(1, samples.shape[2] - frame_offset)
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, 4, generated_t)

    arch, _, _ = detect_model_type(model)
    if arch != "wan":
        raise ValueError(f"Expected a Wan model, got arch='{arch}'.")

    return _apply_prompt_relay_patch(
        model, clip, samples, detect_wan_geometry(model),
        global_prompt, locals_list, parsed_lengths, epsilon,
        frame_offset=frame_offset,
    )


# ----------------------------------------------------------------------------
# WanS2VDirector — Wan2.2 Speech-to-Video
# ----------------------------------------------------------------------------


class WanS2VDirector(io.ComfyNode):
    """Wan2.2-S2V (audio-driven) with Prompt Relay timeline.

    The reference subject image goes on the timeline as the first image segment.
    Audio (from `AudioEncoder` node's output) drives lip-sync. Optional
    control_video / ref_motion provide spatial / motion priors."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanS2VDirector",
            display_name="Wan S2V Director",
            category="WhatDreamsCost",
            description=("Wan2.2-S2V (Speech-to-Video) Director. Audio embedding drives motion; "
                         "place a reference image on the timeline for subject identity."),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True,
                                             tooltip="From an AudioEncoder node — provides the speech embedding for lip-sync."),
                io.Conditioning.Input("negative", optional=True,
                                       tooltip="Optional negative conditioning. Empty negative is built from clip if unconnected."),
                io.Image.Input("control_video", optional=True,
                                tooltip="Optional spatial control video (VAE-encoded and routed to the model)."),
                io.Image.Input("ref_motion", optional=True,
                                tooltip="Optional reference-motion video (up to 73 frames; used as a motion prior)."),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=77, min=1, max=10000, step=4,
                             tooltip="Pixel-frame count. (length - 1) %% 4 == 0 (Wan default 77)."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("frame_offset", default=0, min=0, max=99999, step=1, optional=True,
                             tooltip="Audio frame offset — useful for stitching long clips. Connect from this node's frame_offset output."),
                *_shared_timeline_inputs(),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="frame_offset", tooltip="Updated frame offset for chained S2V calls."),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, global_prompt, width, height, length, batch_size,
                duration_frames, duration_seconds, timeline_data, local_prompts, segment_lengths,
                epsilon=1e-3, frame_rate=24, display_mode="seconds", guide_strength="",
                custom_width=0, custom_height=0, resize_method="maintain aspect ratio",
                divisible_by=16, img_compression=0, use_custom_audio=False, frame_offset=0,
                audio_encoder_output=None, negative=None, control_video=None,
                ref_motion=None) -> io.NodeOutput:

        from comfy_extras.nodes_wan import wan_sound_to_video

        locals_list = _validate_locals(local_prompts)

        # Reference image from timeline (used as ref_image for S2V)
        tgt_w = _snap(custom_width if custom_width > 0 else width, divisible_by)
        tgt_h = _snap(custom_height if custom_height > 0 else height, divisible_by)
        ref_image = _load_first_image_segment(timeline_data, duration_frames, divisible_by,
                                              tgt_w, tgt_h, resize_method)

        # Build empty latent (Wan2.1 geometry: 16ch, /8 spatial, /4 temporal)
        latent_t = ((length - 1) // 4) + 1
        samples = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8],
                              device=comfy.model_management.intermediate_device())

        # Prompt-relay patch
        patched, positive = _common_prompt_relay_setup(
            model, clip, samples, global_prompt, locals_list, segment_lengths, epsilon,
        )

        if negative is None:
            negative = _empty_cond_from_clip(clip)

        # Attach S2V conditioning via ComfyUI's gateway helper
        positive, negative, out_latent, new_frame_offset = wan_sound_to_video(
            positive, negative, vae, width, height, length, batch_size,
            frame_offset=frame_offset,
            ref_image=ref_image,
            audio_encoder_output=audio_encoder_output,
            control_video=control_video,
            ref_motion=ref_motion,
        )

        log.info("[WanS2VDirector] latent_t=%d, frame_offset=%d→%d", latent_t, frame_offset, new_frame_offset)
        return io.NodeOutput(patched, positive, negative, out_latent, int(new_frame_offset), float(frame_rate))


# ----------------------------------------------------------------------------
# WanVaceDirector — Wan2.2 VACE (control video)
# ----------------------------------------------------------------------------


class WanVaceDirector(io.ComfyNode):
    """Wan2.2-VACE (Video-Aware Conditional Encoder) Director.

    Drives generation with a control video (pose / depth / sketch etc.) plus
    optional control masks and a reference image. The control video is split
    into inactive / reactive halves at VAE-encode time, following the standard
    WanVaceToVideo conditioning protocol."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanVaceDirector",
            display_name="Wan VACE Director",
            category="WhatDreamsCost",
            description=("Wan2.2-VACE Director. Combines control_video / control_masks / "
                         "reference_image conditioning with Prompt Relay temporal masking."),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("negative", optional=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("control_masks", optional=True),
                io.Image.Input("reference_image", optional=True,
                                tooltip="Optional reference still. If unconnected, the first timeline image segment is used."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1000.0, step=0.01),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=10000, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                *_shared_timeline_inputs(),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent",
                              tooltip="Number of latent frames at the start that are reference-only (feed into TrimVideoLatent after sampling)."),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, global_prompt, width, height, length, batch_size, strength,
                duration_frames, duration_seconds, timeline_data, local_prompts, segment_lengths,
                epsilon=1e-3, frame_rate=24, display_mode="seconds", guide_strength="",
                custom_width=0, custom_height=0, resize_method="maintain aspect ratio",
                divisible_by=16, img_compression=0, use_custom_audio=False,
                negative=None, control_video=None, control_masks=None,
                reference_image=None) -> io.NodeOutput:

        locals_list = _validate_locals(local_prompts)

        # Fall back to timeline image segment if no explicit reference_image was wired in
        if reference_image is None:
            tgt_w = _snap(custom_width if custom_width > 0 else width, divisible_by)
            tgt_h = _snap(custom_height if custom_height > 0 else height, divisible_by)
            reference_image = _load_first_image_segment(timeline_data, duration_frames, divisible_by,
                                                       tgt_w, tgt_h, resize_method)

        latent_length = ((length - 1) // 4) + 1

        # --- VACE control-video processing (mirrors comfy_extras.nodes_wan.WanVaceToVideo) ---
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1),
                                                       width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        ref_latent = None
        if reference_image is not None:
            ref_img = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1),
                                                  width, height, "bilinear", "center").movedim(1, -1)
            ref_latent = vae.encode(ref_img[:, :, :, :3])
            ref_latent = torch.cat(
                [ref_latent, comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))],
                dim=1,
            )

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        if ref_latent is not None:
            control_video_latent = torch.cat((ref_latent, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if ref_latent is not None:
            mask_pad = torch.zeros_like(mask[:, :ref_latent.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += ref_latent.shape[2]
            trim_latent = ref_latent.shape[2]

        mask = mask.unsqueeze(0)

        # --- Build latent (sized to include the ref padding) ---
        samples = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8],
                              device=comfy.model_management.intermediate_device())

        # --- Prompt-relay patch (skip leading reference frames) ---
        patched, positive = _common_prompt_relay_setup(
            model, clip, samples, global_prompt, locals_list, segment_lengths, epsilon,
            frame_offset=trim_latent,
        )

        if negative is None:
            negative = _empty_cond_from_clip(clip)

        positive = node_helpers.conditioning_set_values(positive, {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_strength": [strength],
        }, append=True)
        negative = node_helpers.conditioning_set_values(negative, {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_strength": [strength],
        }, append=True)

        out_latent = {"samples": samples}
        log.info("[WanVaceDirector] latent_length=%d, trim_latent=%d", latent_length, trim_latent)
        return io.NodeOutput(patched, positive, negative, out_latent, int(trim_latent), float(frame_rate))


# ----------------------------------------------------------------------------
# WanAnimateDirector — Wan2.2 Animate (character animation)
# ----------------------------------------------------------------------------


class WanAnimateDirector(io.ComfyNode):
    """Wan2.2-Animate Director — character animation with pose / face / background controls.

    The character reference image goes on the timeline (first image segment) OR via the
    explicit `reference_image` input. Pose + face videos drive motion + expression."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanAnimateDirector",
            display_name="Wan Animate Director",
            category="WhatDreamsCost",
            description=("Wan2.2-Animate Director. Character animation with pose / face / "
                         "background controls and Prompt Relay temporal masking."),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("negative", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_image", optional=True,
                                tooltip="The subject reference image. If unconnected, the first timeline image segment is used."),
                io.Image.Input("face_video", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.Int.Input("continue_motion_max_frames", default=5, min=1, max=10000, step=4),
                io.Image.Input("background_video", optional=True),
                io.Mask.Input("character_mask", optional=True),
                io.Image.Input("continue_motion", optional=True),
                io.Int.Input("video_frame_offset", default=0, min=0, max=99999, step=1, optional=True),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=77, min=1, max=10000, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                *_shared_timeline_inputs(),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="video_frame_offset"),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, global_prompt, width, height, length, batch_size,
                continue_motion_max_frames, duration_frames, duration_seconds, timeline_data,
                local_prompts, segment_lengths, epsilon=1e-3, frame_rate=24, display_mode="seconds",
                guide_strength="", custom_width=0, custom_height=0,
                resize_method="maintain aspect ratio", divisible_by=16, img_compression=0,
                use_custom_audio=False, video_frame_offset=0,
                negative=None, clip_vision_output=None, reference_image=None, face_video=None,
                pose_video=None, background_video=None, character_mask=None,
                continue_motion=None) -> io.NodeOutput:

        locals_list = _validate_locals(local_prompts)

        # Resolve reference_image from timeline if not explicitly provided
        if reference_image is None:
            tgt_w = _snap(custom_width if custom_width > 0 else width, divisible_by)
            tgt_h = _snap(custom_height if custom_height > 0 else height, divisible_by)
            reference_image = _load_first_image_segment(timeline_data, duration_frames, divisible_by,
                                                       tgt_w, tgt_h, resize_method)
            if reference_image is None:
                reference_image = torch.zeros((1, height, width, 3))

        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        # --- Reference image encode (mirrors WanAnimateToVideo) ---
        image = comfy.utils.common_upscale(reference_image[:length].movedim(-1, 1),
                                            width, height, "area", "center").movedim(1, -1)
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.zeros((1, 4, concat_latent_image.shape[-3],
                            concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                           device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        # --- Continue-motion (carry forward last frames of previous chunk) ---
        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(continue_motion[-length:].movedim(-1, 1),
                                                         width, height, "area", "center").movedim(1, -1)
            image = torch.ones((length, height, width, continue_motion.shape[-1]),
                               device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        # Prompt-relay setup — needs the latent shape to compute tokens_per_frame
        samples = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
                              device=comfy.model_management.intermediate_device())
        # Reference frames sit at the front (trim_latent of them); shift prompt-relay segments past them.
        patched, positive = _common_prompt_relay_setup(
            model, clip, samples, global_prompt, locals_list, segment_lengths, epsilon,
            frame_offset=trim_latent,
        )
        if negative is None:
            negative = _empty_cond_from_clip(clip)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # --- Pose video ---
        if pose_video is not None and pose_video.shape[0] > video_frame_offset:
            pose_video = pose_video[video_frame_offset:]
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1),
                                                    width, height, "area", "center").movedim(1, -1)
            if pose_video.shape[0] < length:
                pose_video = torch.cat((pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

        # --- Face video ---
        if face_video is not None and face_video.shape[0] > video_frame_offset:
            face_video = face_video[video_frame_offset:]
            face_video = comfy.utils.common_upscale(face_video[:length].movedim(-1, 1),
                                                     512, 512, "area", "center") * 2.0 - 1.0
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        # --- Background video ---
        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None and background_video.shape[0] > video_frame_offset:
            background_video = background_video[video_frame_offset:]
            background_video = comfy.utils.common_upscale(background_video[:length].movedim(-1, 1),
                                                          width, height, "area", "center").movedim(1, -1)
            if background_video.shape[0] > ref_images_num:
                image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]

        mask_refmotion = torch.ones(
            (1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=mask.device, dtype=mask.dtype,
        )
        if continue_motion is not None:
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0

        # --- Character mask ---
        if character_mask is not None and (character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1):
            if character_mask.shape[0] == 1:
                character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
            else:
                character_mask = character_mask[video_frame_offset:]
            if character_mask.ndim == 3:
                character_mask = character_mask.unsqueeze(1)
                character_mask = character_mask.movedim(0, 1)
            if character_mask.ndim == 4:
                character_mask = character_mask.unsqueeze(1)
            character_mask = comfy.utils.common_upscale(character_mask[:, :, :length],
                                                        concat_latent_image.shape[-1],
                                                        concat_latent_image.shape[-2],
                                                        "nearest-exact", "center")
            if character_mask.shape[2] > ref_images_num:
                mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]

        concat_latent_image = torch.cat((concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2)

        mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4,
                                              mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        out_latent = {"samples": samples}
        log.info("[WanAnimateDirector] latent_length=%d, trim_latent=%d, video_frame_offset=%d",
                 latent_length + trim_latent, trim_latent, video_frame_offset + length)
        return io.NodeOutput(
            patched, positive, negative, out_latent,
            int(trim_latent), int(max(0, ref_motion_latent_length * 4 - 3)),
            int(video_frame_offset + length), float(frame_rate),
        )


NODE_CLASS_MAPPINGS = {
    "WanDirector": WanDirector,
    "WanS2VDirector": WanS2VDirector,
    "WanVaceDirector": WanVaceDirector,
    "WanAnimateDirector": WanAnimateDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanDirector": "Wan Director",
    "WanS2VDirector": "Wan S2V Director",
    "WanVaceDirector": "Wan VACE Director",
    "WanAnimateDirector": "Wan Animate Director",
}
