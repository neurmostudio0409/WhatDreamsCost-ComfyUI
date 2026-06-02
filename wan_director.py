"""WanDirector — Wan2.2/2.1 timeline node with Prompt Relay + single-pass
multi-keyframe generation (v2.5).

The reference FLF / FMLF workflows get smooth transitions by injecting ALL their
keyframes into ONE latent and sampling ONCE — never by gluing independently-sampled
clips. WanDirector does the same, generalised to an arbitrary number of keyframes:

  - The timeline is grouped into render WINDOWS of <= chunk_frames each. A short
    timeline (the common case) is a single window = one pass.
  - Every keyframe in a window is injected into one latent at its frame position
    (wan_processing.encode_wan_keyframes, the N-keyframe generalisation of
    WanFirstLastFrameToVideo) and the window is sampled ONCE (MoE high→low),
    so motion is continuous across the segments inside it.
  - Within a window, TRUE multi-segment Prompt Relay applies: each segment's prompt
    drives its own latent sub-window (no collapse to a uniform per-clip mask).
  - keyframe_hold pins each reference image across a few frames for stricter adherence.
  - Only timelines longer than one pass span multiple windows; adjacent windows are
    bridged (shared boundary keyframe + carried latent tail) and stitched (1-frame
    seam dropped) into one long latent.

So it absorbs the two KSamplerAdvanced: the graph is just
    loaders → Wan Director → VAEDecode → VideoCombine.

Reuses prompt_relay.py + patches.py (already Wan-aware) + wan_processing.py.
No audio (Wan has none).
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
from .wan_processing import resolve_wan_dims, encode_wan_keyframes, segment_latent_lengths
from .ltx_director import _load_image_tensor  # generic image loader (reused)

log = logging.getLogger(__name__)

WAN_STRIDE = 4


def _snap_len(n):
    """Largest valid Wan length (<= n) on the (len-1) % 4 == 0 grid, min 5."""
    n = max(1, int(n))
    return max(5, ((n - 1) // WAN_STRIDE) * WAN_STRIDE + 1)


def _largest_remainder(weights, total):
    """Distribute integer ``total`` across len(weights) buckets proportional to weights,
    summing EXACTLY to ``total`` (largest-remainder rounding). No frames are lost."""
    s = float(sum(weights)) or 1.0
    exact = [w / s * total for w in weights]
    base = [int(e) for e in exact]
    rem = int(total - sum(base))
    order = sorted(range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True)
    for k in range(max(0, rem)):
        base[order[k % len(order)]] += 1
    return base


def _parse_relay_segments(local_prompts, segment_lengths, timeline_data, duration_frames):
    """Prompt-relay segments — (prompts, pixel_lengths) — preferring the JS-computed
    ``local_prompts`` (|-separated) + ``segment_lengths`` (,-separated). That is the exact
    contract the working LTXDirector consumes: contiguous regions, gaps merged, padded to
    the duration. Falls back to timeline_data segment prompts/lengths if those are empty."""
    prompts = [p.strip() for p in local_prompts.split("|")] if (local_prompts or "").strip() else []
    lens = []
    if (segment_lengths or "").strip():
        for x in segment_lengths.split(","):
            x = x.strip()
            if x:
                try:
                    lens.append(max(1, int(float(x))))
                except ValueError:
                    pass
    if prompts and lens and len(prompts) == len(lens):
        return prompts, lens

    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
    except (ValueError, TypeError):
        tdata = {}
    segs = []
    for s in tdata.get("segments", []):
        if s.get("type") in ("temp", "ghost", "audio"):
            continue
        st = int(s.get("start", 0))
        if st >= duration_frames:
            continue
        segs.append((st, max(1, int(s.get("length", 1))), (s.get("prompt") or "").strip()))
    segs.sort(key=lambda x: x[0])
    return [p for (_s, _l, p) in segs], [l for (_s, l, _p) in segs]


def _parse_keyframes(timeline_data, duration_frames):
    """Image keyframes ``[(start_frame, raw_segment)]`` from the timeline — image-bearing
    visual segments only (skips temp/ghost/audio and prompt-only segments)."""
    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
    except (ValueError, TypeError):
        return []
    out = []
    for s in tdata.get("segments", []):
        if s.get("type") in ("temp", "ghost", "audio"):
            continue
        if not (s.get("imageFile") or s.get("imageB64")):
            continue
        st = int(s.get("start", 0))
        if st >= duration_frames:
            continue
        out.append((st, s))
    out.sort(key=lambda x: x[0])
    return out


def _plan_windows(prompts, seg_lens, keyframes, total_len, duration_frames, chunk_max):
    """Cut the timeline into render WINDOWS of <= chunk_max frames, PRESERVING total length
    (a span longer than one pass is SPLIT across windows, never truncated). The prompt-relay
    regions and the image keyframes are mapped onto each window independently.

    Returns windows: ``{"length", "segments": [(prompt, local_len)], "keyframes": [(local_frame, raw)]}``.
    A window's frame 0 is a keyframe at local 0 (segment boundary) or — in later windows —
    the previous window's carried latent tail (decided by the caller).
    """
    chunk_max = max(5, int(chunk_max))

    # Prompt regions scaled to exactly total_len (no rounding drift).
    glens = _largest_remainder([max(1, l) for l in seg_lens], total_len) if seg_lens else [total_len]
    rprompts = list(prompts) if prompts else [""]
    if len(rprompts) != len(glens):
        rprompts = (rprompts + [""] * len(glens))[:len(glens)]
    regions, c = [], 0
    for p, L in zip(rprompts, glens):
        if L <= 0:
            continue
        regions.append((c, c + L, p))
        c += L
    if not regions:
        regions = [(0, total_len, "")]

    # Keyframe positions mapped from timeline (duration_frames) space into total_len space.
    dur = max(1, int(duration_frames))
    kf_pos = sorted(
        (max(0, min(int(round(st * total_len / dur)), total_len - 1)), raw)
        for (st, raw) in keyframes
    )

    windows, pos = [], 0
    while pos < total_len:
        remaining = total_len - pos
        wlen = remaining if remaining <= chunk_max else _snap_len(chunk_max)
        wlen = max(1, min(wlen, remaining))
        wstart, wend = pos, pos + wlen
        wsegs = [(p, min(b, wend) - max(a, wstart))
                 for (a, b, p) in regions if min(b, wend) > max(a, wstart)]
        wkfs = [(f - wstart, raw) for (f, raw) in kf_pos if wstart <= f < wend]
        windows.append({"length": wlen, "segments": wsegs or [("", wlen)], "keyframes": wkfs})
        pos = wend
    return windows


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
    """Wan timeline editor with Prompt Relay and single-pass multi-keyframe generation.
    Drop an image on each timeline segment; ALL keyframes that fit one pass are injected
    into a single latent and sampled once (smooth transitions, like FLF/FMLF), with each
    segment's prompt driving its own time window. Only timelines longer than one pass span
    multiple bridged windows, stitched into one latent — no manual chaining."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDirector",
            display_name="Wan Director",
            category="WhatDreamsCost",
            description=(
                "Wan2.2/2.1 timeline editor with Prompt Relay. The timeline duration sets the "
                "video length; all keyframes that fit one pass are injected into a single latent "
                "and sampled once for smooth transitions (longer timelines span bridged windows). "
                "Samples internally (MoE high→low) and outputs one long latent — feed it to VAEDecode."
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
                                     "Longer than chunk_frames spans multiple bridged windows automatically."),
                io.Int.Input("batch_size", default=1, min=1, max=4096,
                             tooltip="Keep at 1 for multi-window (long-video) generation."),
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
                             tooltip="Sampling steps per window."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1,
                               tooltip="CFG scale. 1.0 with a lightx2v-style distill LoRA."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Base noise seed (each window uses seed + window index)."),
                io.Int.Input("chunk_frames", default=81, min=5, max=10000, step=4,
                             tooltip="Max frames per single Wan pass = one render window (~81 for 14B). All "
                                     "keyframes within a window share one continuous pass; raise it to keep more "
                                     "segments in one smooth window (slower, more VRAM)."),
                io.Int.Input("moe_boundary", default=4, min=1, max=200,
                             tooltip="Step at which sampling switches from the high-noise to the low-noise model "
                                     "(ignored if model_low is unconnected)."),
                # --- optional widgets ---
                io.Combo.Input("i2v_backend", options=["native", "fmlf"], default="native", optional=True,
                               tooltip="Reserved. The generator now uses native N-keyframe injection "
                                       "(encode_wan_keyframes) for single-pass multi-keyframe transitions; this "
                                       "selector is kept for widget-order compatibility and a future fmlf backend."),
                io.Float.Input("frame_rate", default=16, min=1, max=240, step=1, optional=True,
                               tooltip="Playback fps. Each chained piece spans chunk_frames÷frame_rate seconds, so at "
                                       "the default 81÷16≈5s per piece (24 fps would be ~3.4s). The timeline editor "
                                       "also converts its seconds<->frames with THIS value, so total duration still "
                                       "matches what you set. Pair with RIFE downstream to interpolate back up for smoothness."),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.Int.Input("divisible_by", default=16, min=1, max=256, step=1, optional=True,
                             tooltip="Snap auto-detected dimensions to a multiple of this (Wan2.1: 16, 2.2-5B: 32)."),
                io.Int.Input("max_side", default=832, min=0, max=8192, step=16, optional=True,
                             tooltip="Cap the longest side (keeps aspect ratio). MAIN speed lever — Wan is trained "
                                     "near 480p. 832 ≈ 480p (fast), 1280 ≈ 720p (slower, sharper), 0 = no cap (uses "
                                     "the image's native size; can be very slow for large images)."),
                io.String.Input("guide_strength", default="", optional=True,
                                tooltip="Unused for Wan; kept for shared timeline-JS compatibility."),
                io.Int.Input("keyframe_hold", default=5, min=1, max=81, step=1, optional=True,
                             tooltip="How many frames to PIN each keyframe (reference image) as known. "
                                     "Higher = stricter adherence to the reference image, at the cost of a "
                                     "brief hold/freeze on that keyframe. Native Wan pins ((hold-1)//4)+1 "
                                     "latent frames: 1 = single frame (loosest), 5 ≈ pin 2 latent frames, "
                                     "9 ≈ 3. On a first-last (FLF) chunk it is capped to half the chunk so "
                                     "there is room to move between the two keyframes."),
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
                i2v_backend="native", frame_rate=16, display_mode="seconds", divisible_by=16,
                max_side=832, guide_strength="", keyframe_hold=5, model_low=None, clip_vision=None,
                clip_vision_start=None, clip_vision_end=None) -> io.NodeOutput:

        arch, patch_size, _ = detect_model_type(model_high)
        if arch != "wan":
            raise ValueError(
                f"WanDirector expects a Wan diffusion model on model_high, got arch='{arch}'. "
                "Use LTXDirector for LTX models."
            )
        if vae is None:
            raise ValueError("WanDirector: a Wan VAE is required (encodes images, decodes seams).")

        # --- Parse the timeline. Prompt-relay segments come from the JS-computed
        #     local_prompts/segment_lengths (the proven LTXDirector contract); image
        #     keyframes are read separately from timeline_data. The two are decoupled. ---
        prompts, seg_lens = _parse_relay_segments(local_prompts, segment_lengths,
                                                  timeline_data, duration_frames)
        keyframes = _parse_keyframes(timeline_data, duration_frames)
        if not keyframes and not any((p or "").strip() for p in prompts):
            raise ValueError("WanDirector: the timeline is empty. Add at least one segment "
                             "(image keyframe and/or prompt).")

        # --- Total length (0 = follow timeline). Prefer the segment-length sum so the prompt
        #     regions map exactly; else the timeline duration. Length is PRESERVED (windows
        #     split, never truncate). ---
        if length and int(length) > 0:
            total_len = _snap_len(int(length))
        else:
            total_len = _snap_len(sum(seg_lens) if seg_lens else int(duration_frames))
        first_image = _load_image_tensor(keyframes[0][1]) if keyframes else None
        tgt_w, tgt_h = resolve_wan_dims(first_image, width, height, divisible_by, max_side)

        # --- Plan render windows: each window is ONE multi-keyframe single pass; only a
        #     timeline longer than one Wan pass spans multiple (bridged) windows. ---
        windows = _plan_windows(prompts, seg_lens, keyframes, total_len, duration_frames, chunk_frames)
        n = len(windows)
        log.info("[WanDirector] %d prompt-seg, %d keyframe(s) -> %d window(s), total~%d frames, %dx%d",
                 len(prompts), len(keyframes), n, total_len, tgt_w, tgt_h)

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative_prompt or ""))
        raw_tokenizer = get_raw_tokenizer(clip)

        hold = max(1, int(keyframe_hold))
        stitched = None
        prev_tail = None  # decoded tail of the previous window (latent-continuity bridge)
        for wi, win in enumerate(windows):
            w_len = win["length"]

            # Keyframes injected into this single pass. Each: (local_frame, image, is_reference).
            # If no keyframe sits at this window's start, a later window carries the previous
            # window's decoded tail at frame 0 for continuity (a continuation, not "held").
            kf = []
            wkfs = win["keyframes"]
            if wi > 0 and prev_tail is not None and not any(int(lf) == 0 for (lf, _r) in wkfs):
                kf.append((0, prev_tail, False))
            for (lf, raw) in wkfs:
                kf.append((int(lf), _load_image_tensor(raw), True))

            # CLIP-Vision per keyframe (encode the single frame; the semantic anchor).
            cv_outputs = []
            for idx, (fr, img, is_ref) in enumerate(kf):
                if clip_vision is not None and img is not None:
                    cv_outputs.append(clip_vision.encode_image(img[:1], crop=True))
                elif wi == 0 and idx == 0 and clip_vision_start is not None:
                    cv_outputs.append(clip_vision_start)
                else:
                    cv_outputs.append(None)

            # Hold reference keyframes (pins more latent frames -> stricter adherence); a
            # carried tail stays one frame. encode_wan_keyframes does the in-latent repeat.
            inject = [(fr, img, hold if is_ref else 1) for (fr, img, is_ref) in kf]

            # True multi-segment Prompt Relay: each segment's prompt drives its own latent
            # sub-window WITHIN this single pass (no per-clip collapse to a uniform mask).
            w_prompts = [p for (p, _l) in win["segments"]]
            w_seg_pix = [l for (_p, l) in win["segments"]]
            full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, w_prompts)
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

            positive, neg_c, latent, latent_frames = encode_wan_keyframes(
                positive, negative, vae, tgt_w, tgt_h, w_len, batch_size,
                inject, clip_vision_outputs=cv_outputs,
            )

            samples = latent["samples"]
            tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
            eff = segment_latent_lengths(w_seg_pix, latent_frames) or \
                distribute_segment_lengths(len(w_prompts), latent_frames, None)
            mask_fn = create_mask_fn(build_segments(token_ranges, eff, epsilon, None),
                                     tokens_per_frame, latent_frames)
            ph = model_high.clone()
            apply_patches(ph, arch, mask_fn)
            pl = None
            if model_low is not None:
                pl = model_low.clone()
                apply_patches(pl, arch, mask_fn)

            sampled = _moe_sample(ph, pl, int(seed) + wi, int(steps), float(cfg),
                                  sampler_name, scheduler, positive, neg_c, latent, moe_boundary)
            sout = sampled["samples"]

            # Stitch (drop the 1-frame seam each window shares with the previous tail).
            stitched = sout if stitched is None else torch.cat([stitched, sout[:, :, 1:]], dim=2)
            log.info("[WanDirector] window %d/%d: %d frames, %d segment(s), %d keyframe(s), hold=%d -> %s",
                     wi + 1, n, w_len, len(w_prompts), len(inject), hold, tuple(stitched.shape))

            if wi < n - 1:
                prev_tail = _decode_tail_start(vae, sout)

        return io.NodeOutput({"samples": stitched}, float(frame_rate))
