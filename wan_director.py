"""WanDirector — Wan2.2/2.1 timeline node with Prompt Relay + internal chunked
long-video generation (v3).

Wan I2V is a single-pass model (~5s / ~81 frames). To give the LTX-Director
experience — one timeline → an arbitrarily long video — WanDirector does the
segmentation ITSELF, on the backend, KEYFRAME-driven:

  - Each timeline segment is a keyframe region: its image anchors that part of
    the video, its prompt drives that part.
  - A segment is sub-split into <= chunk_frames pieces. Its first piece starts
    from the segment's image (the keyframe); later pieces chain from the previous
    decoded tail; the last piece morphs to the NEXT segment's image via
    first-last-frame (FLF) — so the video passes through every keyframe in order.
  - Each piece is generated as a Wan I2V/FLF (MoE high→low sampling, internal).
  - All piece latents are stitched (1-frame seam dropped) into one long latent.

So it absorbs the two KSamplerAdvanced: the graph is just
    loaders → Wan Director → VAEDecode → VideoCombine.

Reuses prompt_relay.py + patches.py (already Wan-aware) + wan_processing.py
(delegates latent/I2V conditioning to native Wan nodes). No audio (Wan has none).
"""
import json
import logging

import torch
import comfy.samplers

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)
from .patches import detect_model_type, apply_patches
from .wan_processing import resolve_wan_dims, encode_wan_i2v, segment_latent_lengths
from .ltx_director import _load_image_tensor  # generic image loader (reused)

log = logging.getLogger(__name__)

WAN_STRIDE = 4


def _snap_len(n):
    """Largest valid Wan length (<= n) on the (len-1) % 4 == 0 grid, min 5."""
    n = max(1, int(n))
    return max(5, ((n - 1) // WAN_STRIDE) * WAN_STRIDE + 1)


def _parse_timeline(timeline_data, duration_frames):
    """All timeline segments (in order) with their prompt, length, and image (if any).
    Each segment is a KEYFRAME region: its image anchors that part of the video."""
    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
    except (ValueError, TypeError):
        return []
    out = []
    for s in tdata.get("segments", []):
        if int(s.get("start", 0)) >= duration_frames:
            continue
        out.append({
            "start": int(s.get("start", 0)),
            "length": max(1, int(s.get("length", 1))),
            "prompt": (s.get("prompt") or "").strip(),
            "raw": s,
            "has_img": bool(s.get("imageFile") or s.get("imageB64")),
        })
    out.sort(key=lambda x: x["start"])
    return out


def _plan_pieces(seg_len, chunk_max):
    """Split one segment's frame length into evenly-sized pieces of <= chunk_max
    (valid Wan lengths) — avoids tiny remainder pieces that make abrupt FLF morphs."""
    seg_len = max(1, int(seg_len))
    chunk_max = max(5, int(chunk_max))
    if seg_len <= chunk_max:
        return [_snap_len(seg_len)]
    n = (seg_len + chunk_max - 1) // chunk_max  # ceil
    base = max(5, seg_len // n)
    return [_snap_len(base) for _ in range(n)]


def _build_chunk_specs(segs, total_len, chunk_max):
    """Turn timeline segments into a flat list of chunk specs.

    Each segment's image anchors its first piece (the keyframe). Within a segment
    longer than chunk_max, later pieces chain from the previous decoded tail. The
    LAST piece of a segment morphs to the NEXT segment's image via first-last-frame
    (FLF), so the video passes through every keyframe in order.
    """
    sum_len = sum(s["length"] for s in segs) or 1
    scale = total_len / sum_len
    specs = []
    for k, s in enumerate(segs):
        seg_len = max(_snap_len(1), int(round(s["length"] * scale)))
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        next_img_raw = nxt["raw"] if (nxt and nxt["has_img"]) else None
        pieces = _plan_pieces(seg_len, chunk_max)
        for pi, plen in enumerate(pieces):
            specs.append({
                "prompt": s["prompt"],
                "length": plen,
                "start_raw": s["raw"] if (pi == 0 and s["has_img"]) else None,  # None -> chain from tail
                "end_raw": next_img_raw if pi == len(pieces) - 1 else None,     # FLF morph to next keyframe
            })
    return specs


def _moe_sample(model_high, model_low, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, boundary):
    """Two-stage MoE sampling (high-noise then low-noise), mirroring the
    KSamplerAdvanced pair. Single model if model_low is None."""
    from nodes import common_ksampler
    boundary = max(1, min(int(boundary), int(steps) - 1)) if model_low is not None else int(steps)
    if model_low is None:
        return common_ksampler(model_high, seed, steps, cfg, sampler_name, scheduler,
                               positive, negative, latent,
                               disable_noise=False, start_step=0, last_step=steps,
                               force_full_denoise=True)[0]
    hi = common_ksampler(model_high, seed, steps, cfg, sampler_name, scheduler,
                         positive, negative, latent,
                         disable_noise=False, start_step=0, last_step=boundary,
                         force_full_denoise=False)[0]
    return common_ksampler(model_low, seed, steps, cfg, sampler_name, scheduler,
                           positive, negative, hi,
                           disable_noise=True, start_step=boundary, last_step=steps,
                           force_full_denoise=True)[0]


def _decode_tail_start(vae, samples):
    """Decode the last latent frame -> a single start-image tensor [1, H, W, C]
    for the next chunk's I2V keyframe."""
    decoded = vae.decode(samples[:, :, -1:].contiguous())
    if decoded.dim() == 5:  # [B, T, H, W, C] -> [T, H, W, C]
        decoded = decoded[0]
    return decoded[-1:]


class WanDirector(io.ComfyNode):
    """Wan timeline editor with Prompt Relay and internal long-video generation.
    Drop an image at the start of the timeline for I2V (start + end = FLF); prompt
    segments drive the content along the timeline. Long timelines are generated as
    chained <=chunk_frames chunks and stitched into one latent — no manual chaining."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDirector",
            display_name="Wan Director",
            category="WhatDreamsCost",
            description=(
                "Wan2.2/2.1 timeline editor with Prompt Relay. The timeline duration sets the "
                "video length; longer-than-one-pass timelines are generated as chained chunks and "
                "stitched automatically (Wan I2V is single-pass ~5s). Samples internally (MoE "
                "high→low) and outputs one long latent — feed it straight to VAEDecode."
            ),
            inputs=[
                io.Model.Input("model_high", tooltip="Wan diffusion model. For MoE 14B this is the high-noise model."),
                io.Model.Input("model_low", optional=True,
                               tooltip="Optional — Wan2.2 MoE 14B low-noise model (second sampling stage)."),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="Wan VAE — encodes start/end images, decodes chunk seams, sizes the latent."),
                io.ClipVision.Input("clip_vision", optional=True,
                                    tooltip="Strongly recommended for I2V: a CLIP-Vision model (clip_vision_h). The "
                                            "Director encodes each chunk's start image with it — the semantic anchor "
                                            "that keeps the output faithful to the input image."),
                io.ClipVisionOutput.Input("clip_vision_start", optional=True,
                                          tooltip="Optional pre-encoded CLIP-Vision output of the start image (overrides internal)."),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True,
                                          tooltip="Optional pre-encoded CLIP-Vision output of the end image (FLF only)."),
                # --- required timeline widgets (names mirror LTX Director for the shared JS) ---
                io.String.Input("global_prompt", multiline=True, default="",
                                tooltip="Conditions the entire video; anchors persistent characters / scene."),
                io.String.Input("global_negative_prompt", multiline=True, default="",
                                tooltip="Global negative prompt. Toggle 'Use Global Negative Prompt' in the timeline "
                                        "settings to show/edit it. Empty = empty negative."),
                io.Int.Input("width", default=0, min=0, max=8192, step=16,
                             tooltip="0 = auto-detect from the start image (snapped to divisible_by). Set for T2V."),
                io.Int.Input("height", default=0, min=0, max=8192, step=16,
                             tooltip="0 = auto-detect from the start image (snapped to divisible_by). Set for T2V."),
                io.Int.Input("length", default=0, min=0, max=100000, step=4,
                             tooltip="Total video length in frames. 0 = follow the timeline duration (like LTX). "
                                     "Longer than chunk_frames is split into chained chunks automatically."),
                io.Int.Input("batch_size", default=1, min=1, max=4096,
                             tooltip="Keep at 1 for chained (multi-chunk) generation."),
                io.Int.Input("duration_frames", default=120, min=1, max=100000, step=1,
                             tooltip="Timeline length in frames (drives total length when length = 0)."),
                io.Float.Input("duration_seconds", default=5.0, min=0.1, max=10000.0, step=0.01,
                               tooltip="Total timeline duration in seconds (display only)."),
                io.String.Input("timeline_data", default="",
                                tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand)."),
                io.String.Input("local_prompts", multiline=True, default="",
                                tooltip="Auto-populated from the timeline editor (pipe-separated segment prompts)."),
                io.String.Input("segment_lengths", default="",
                                tooltip="Auto-populated from the timeline editor (frame counts per segment)."),
                io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                               tooltip="Prompt Relay penalty decay (paper default 0.001)."),
                io.Int.Input("steps", default=8, min=1, max=200,
                             tooltip="Sampling steps per chunk."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1,
                               tooltip="CFG scale. 1.0 with a lightx2v-style distill LoRA."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Base noise seed (each chunk uses seed + chunk index)."),
                io.Int.Input("chunk_frames", default=81, min=5, max=10000, step=4,
                             tooltip="Max frames generated per single Wan pass before chaining (~81 for 14B)."),
                io.Int.Input("moe_boundary", default=4, min=1, max=200,
                             tooltip="Step at which sampling switches from the high-noise to the low-noise model "
                                     "(ignored if model_low is unconnected)."),
                # --- optional widgets ---
                io.Combo.Input("i2v_backend", options=["native", "fmlf"], default="native", optional=True,
                               tooltip="I2V conditioning backend. native = WanImageToVideo/FLF (no deps). "
                                       "fmlf = Wan22FMLF WanAdvancedI2V (falls back to native if absent)."),
                io.Float.Input("frame_rate", default=24, min=1, max=240, step=1, optional=True),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.Int.Input("divisible_by", default=16, min=1, max=256, step=1, optional=True,
                             tooltip="Snap auto-detected dimensions to a multiple of this (Wan2.1: 16, 2.2-5B: 32)."),
                io.Int.Input("max_side", default=832, min=0, max=8192, step=16, optional=True,
                             tooltip="Cap the longest side (keeps aspect ratio). MAIN speed lever — Wan is trained "
                                     "near 480p. 832 ≈ 480p (fast), 1280 ≈ 720p (slower, sharper), 0 = no cap (uses "
                                     "the image's native size; can be very slow for large images)."),
                io.String.Input("guide_strength", default="", optional=True,
                                tooltip="Unused for Wan; kept for shared timeline-JS compatibility."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent",
                                 tooltip="The full (stitched) video latent. Feed straight to VAEDecode."),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model_high, clip, vae, global_prompt, global_negative_prompt, width, height,
                length, batch_size, duration_frames, duration_seconds, timeline_data, local_prompts,
                segment_lengths, epsilon=1e-3, steps=8, cfg=1.0, sampler_name="euler",
                scheduler="simple", seed=0, chunk_frames=81, moe_boundary=4,
                i2v_backend="native", frame_rate=24, display_mode="seconds", divisible_by=16,
                max_side=832, guide_strength="", model_low=None, clip_vision=None,
                clip_vision_start=None, clip_vision_end=None) -> io.NodeOutput:

        arch, patch_size, _ = detect_model_type(model_high)
        if arch != "wan":
            raise ValueError(
                f"WanDirector expects a Wan diffusion model on model_high, got arch='{arch}'. "
                "Use LTXDirector for LTX models."
            )
        if vae is None:
            raise ValueError("WanDirector: a Wan VAE is required (encodes images, decodes seams).")

        # --- Parse timeline segments (each = a keyframe region: image + prompt + length) ---
        segs = _parse_timeline(timeline_data, duration_frames)
        if not segs:
            raise ValueError("WanDirector: the timeline is empty. Add at least one segment (image + prompt).")
        for s in segs:
            if not s["prompt"]:
                raise ValueError("There is a segment on the timeline missing a prompt!")

        # --- Total length (0 = follow timeline), then dimensions from the first keyframe ---
        total_len = int(length) if length and int(length) > 0 else _snap_len(int(duration_frames))
        total_len = _snap_len(total_len)
        first_img_seg = next((s for s in segs if s["has_img"]), None)
        first_image = _load_image_tensor(first_img_seg["raw"]) if first_img_seg else None
        tgt_w, tgt_h = resolve_wan_dims(first_image, width, height, divisible_by, max_side)

        # --- Build chunk specs: each segment's image anchors its region; FLF morph to next keyframe ---
        specs = _build_chunk_specs(segs, total_len, chunk_frames)
        n = len(specs)
        log.info("[WanDirector] %d segment(s) -> %d chunk(s), total~%d frames, %dx%d, backend=%s",
                 len(segs), n, total_len, tgt_w, tgt_h, i2v_backend)

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative_prompt or ""))
        raw_tokenizer = get_raw_tokenizer(clip)

        stitched = None
        prev_tail = None  # decoded tail of the previous chunk (for chaining within a segment)
        for si, spec in enumerate(specs):
            # Start image = this segment's keyframe (first piece) or the previous tail (chained).
            start_image = _load_image_tensor(spec["start_raw"]) if spec["start_raw"] is not None else prev_tail
            # End image = next segment's keyframe (FLF morph), only on a segment's last piece.
            end_image = _load_image_tensor(spec["end_raw"]) if spec["end_raw"] is not None else None

            cv_start = cv_end = None
            if clip_vision is not None:
                if start_image is not None:
                    cv_start = clip_vision.encode_image(start_image, crop=True)
                if end_image is not None:
                    cv_end = clip_vision.encode_image(end_image, crop=True)
            elif si == 0:
                cv_start, cv_end = clip_vision_start, clip_vision_end

            # One prompt per chunk (its segment's). Relay reduces to a uniform mask here.
            full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, [spec["prompt"]])
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

            positive, neg_c, latent, latent_frames = encode_wan_i2v(
                positive, negative, vae, tgt_w, tgt_h, spec["length"], batch_size,
                start_image=start_image, end_image=end_image,
                clip_vision_start=cv_start, clip_vision_end=cv_end,
                backend=i2v_backend,
            )

            samples = latent["samples"]
            tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
            eff = distribute_segment_lengths(1, latent_frames, None)
            mask_fn = create_mask_fn(build_segments(token_ranges, eff, epsilon, None),
                                     tokens_per_frame, latent_frames)
            ph = model_high.clone()
            apply_patches(ph, arch, mask_fn)
            pl = None
            if model_low is not None:
                pl = model_low.clone()
                apply_patches(pl, arch, mask_fn)

            sampled = _moe_sample(ph, pl, int(seed) + si, int(steps), float(cfg),
                                  sampler_name, scheduler, positive, neg_c, latent, moe_boundary)
            sout = sampled["samples"]

            # Stitch (drop the 1-frame seam each chunk shares with the previous frame).
            stitched = sout if stitched is None else torch.cat([stitched, sout[:, :, 1:]], dim=2)
            log.info("[WanDirector] chunk %d/%d: %d frames, kf=%s flf=%s -> stitched %s",
                     si + 1, n, spec["length"], spec["start_raw"] is not None,
                     end_image is not None, tuple(stitched.shape))

            if si < n - 1:
                prev_tail = _decode_tail_start(vae, sout)

        return io.NodeOutput({"samples": stitched}, float(frame_rate))
