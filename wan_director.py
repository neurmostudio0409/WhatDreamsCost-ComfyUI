"""WanDirector — Wan2.2/2.1 timeline node with Prompt Relay + internal chunked
long-video generation (v3).

Wan I2V is a single-pass model (~5s / ~81 frames). To give the LTX-Director
experience — one timeline → an arbitrarily long video — WanDirector does the
segmentation ITSELF, on the backend:

  - The timeline duration sets the total length.
  - The node splits it into <= chunk_frames pieces.
  - Each chunk is generated as a Wan I2V (MoE high→low sampling, internal), with
    its prompt taken from the timeline region it covers (Prompt Relay per chunk).
  - Chunk N+1's start image = chunk N's last decoded frame (seamless continuation).
  - All chunk latents are stitched (1-frame seam dropped) into one long latent.

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


def _extract_image_segments(timeline_data, duration_frames):
    """Image segments from the timeline JSON, sorted by start (first = start frame,
    last = end frame for FLF)."""
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


def _segment_spans(locals_list, segment_lengths_str, total_len):
    """Reconstruct each prompt segment's [start, end) span in frames, scaled to
    total_len. Falls back to an even split when lengths are missing/mismatched."""
    n = len(locals_list)
    lens = []
    if segment_lengths_str and segment_lengths_str.strip():
        lens = [int(float(x.strip())) for x in segment_lengths_str.split(",") if x.strip()]
    if len(lens) != n or sum(lens) <= 0:
        base = max(1, total_len // n)
        lens = [base] * n
        lens[-1] += total_len - base * n
    scale = total_len / max(1, sum(lens))
    spans = []
    cur = 0.0
    for p, l in zip(locals_list, lens):
        start = cur
        cur += l * scale
        spans.append((p, start, cur))
    return spans


def _plan_chunks(total_len, chunk_max):
    """Split total_len frames into chunks of <= chunk_max (each a valid Wan length).
    Consecutive chunks share 1 frame (chunk N+1 starts on chunk N's last frame).
    Returns (chunk_lengths, windows) where windows[i] = (start, end) unique-frame
    range of chunk i in total_len space."""
    total_len = _snap_len(total_len)
    cmax = _snap_len(min(chunk_max, total_len))
    chunks, windows = [], []
    produced = 0
    while produced < total_len and len(chunks) < 256:
        if not chunks:
            clen = _snap_len(min(cmax, total_len))
            new = clen
        else:
            want = (total_len - produced) + 1  # +1 shared start frame
            clen = _snap_len(min(cmax, want))
            new = clen - 1
            if new <= 0:
                break
        windows.append((produced, produced + new))
        produced += new
        chunks.append(clen)
    return chunks, windows


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

        # --- Validate prompt segments ---
        locals_list = [p.strip() for p in (local_prompts or "").split("|")]
        for p in locals_list:
            if not p:
                raise ValueError("There is a segment on the timeline missing a prompt!")
        if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
            raise ValueError("At least one local prompt is required.")

        arch, patch_size, _ = detect_model_type(model_high)
        if arch != "wan":
            raise ValueError(
                f"WanDirector expects a Wan diffusion model on model_high, got arch='{arch}'. "
                "Use LTXDirector for LTX models."
            )
        if vae is None:
            raise ValueError("WanDirector: a Wan VAE is required (encodes images, decodes seams).")

        # --- Total length (0 = follow timeline), then dimensions ---
        total_len = int(length) if length and int(length) > 0 else _snap_len(int(duration_frames))
        total_len = _snap_len(total_len)

        img_segs = _extract_image_segments(timeline_data, duration_frames)
        timeline_start = _load_image_tensor(img_segs[0]) if len(img_segs) >= 1 else None
        timeline_end = _load_image_tensor(img_segs[-1]) if len(img_segs) >= 2 else None
        tgt_w, tgt_h = resolve_wan_dims(timeline_start, width, height, divisible_by, max_side)

        # --- Plan chunks + map each chunk to its timeline prompts ---
        spans = _segment_spans(locals_list, segment_lengths, total_len)
        chunk_lens, windows = _plan_chunks(total_len, chunk_frames)
        n_chunks = len(chunk_lens)
        log.info("[WanDirector] total=%d frames, %d chunk(s) %s, %dx%d, backend=%s",
                 total_len, n_chunks, chunk_lens, tgt_w, tgt_h, i2v_backend)

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative_prompt or ""))
        raw_tokenizer = get_raw_tokenizer(clip)

        stitched = None
        prev_start = None  # decoded tail of the previous chunk
        for ci, (clen, (ws, we)) in enumerate(zip(chunk_lens, windows)):
            is_last = ci == n_chunks - 1
            start_image = timeline_start if ci == 0 else prev_start
            end_image = timeline_end if (is_last and timeline_end is not None) else None

            # Per-chunk CLIP-Vision encode of the start image.
            cv_start, cv_end = clip_vision_start, clip_vision_end
            if clip_vision is not None:
                if start_image is not None and (ci != 0 or cv_start is None):
                    cv_start = clip_vision.encode_image(start_image, crop=True)
                if end_image is not None and cv_end is None:
                    cv_end = clip_vision.encode_image(end_image, crop=True)

            # Prompts covering this chunk's timeline window -> per-chunk relay.
            chunk_locals, chunk_seg_px = [], []
            for prompt, ss, se in spans:
                ov = min(se, we) - max(ss, ws)
                if ov > 0:
                    chunk_locals.append(prompt)
                    chunk_seg_px.append(int(round(ov)))
            if not chunk_locals:  # safety: nearest span by midpoint
                mid = (ws + we) / 2
                prompt = min(spans, key=lambda s: abs((s[1] + s[2]) / 2 - mid))[0]
                chunk_locals, chunk_seg_px = [prompt], [we - ws]

            full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, chunk_locals)
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

            # Build the chunk's Wan latent + I2V/FLF/T2V conditioning.
            positive, neg_c, latent, latent_frames = encode_wan_i2v(
                positive, negative, vae, tgt_w, tgt_h, clen, batch_size,
                start_image=start_image, end_image=end_image,
                clip_vision_start=cv_start, clip_vision_end=cv_end,
                backend=i2v_backend,
            )

            # Prompt Relay temporal mask for this chunk.
            samples = latent["samples"]
            tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
            parsed = segment_latent_lengths(chunk_seg_px, latent_frames)
            eff = distribute_segment_lengths(len(chunk_locals), latent_frames, parsed)
            mask_fn = create_mask_fn(build_segments(token_ranges, eff, epsilon, None),
                                     tokens_per_frame, latent_frames)

            ph = model_high.clone()
            apply_patches(ph, arch, mask_fn)
            pl = None
            if model_low is not None:
                pl = model_low.clone()
                apply_patches(pl, arch, mask_fn)

            # Internal MoE sampling for this chunk.
            sampled = _moe_sample(ph, pl, int(seed) + ci, int(steps), float(cfg),
                                  sampler_name, scheduler, positive, neg_c, latent, moe_boundary)
            s = sampled["samples"]

            # Stitch (drop the 1-frame seam shared with the previous chunk).
            stitched = s if stitched is None else torch.cat([stitched, s[:, :, 1:]], dim=2)
            log.info("[WanDirector] chunk %d/%d: %d frames, prompts=%d -> stitched %s",
                     ci + 1, n_chunks, clen, len(chunk_locals), tuple(stitched.shape))

            # Decode this chunk's tail as the next chunk's start image.
            if not is_last:
                prev_start = _decode_tail_start(vae, s)

        return io.NodeOutput({"samples": stitched}, float(frame_rate))
