"""WanDirector — Wan2.2/2.1 timeline node, segment-aligned first-last-frame (FLF) chaining (v4).

Both reference workflows ("FLframe transition", "WAN 2.2 Smooth v5") build a long video as a
chain of FLF clips: each clip morphs from a start image to an end image, and the next clip
starts from the previous clip's LAST frame, so the segments connect (no independent clips).
WanDirector does exactly that from the timeline:

  - Each keyframe is a hard boundary; segment i = one FLF clip morphing keyframe_i ->
    keyframe_{i+1} (the last segment, and any span longer than chunk_frames, continue as I2V).
  - Continuity is carried in LATENT space (the previous clip's last latent frame seeds the
    next clip — no decode/re-encode round-trip, so colour stays stable); the keyframe images
    are the FLF morph TARGETS. Each clip is sampled MoE high->low (encode_wan_keyframes =
    native FLF conditioning), and the per-clip latents are stitched into one long latent.
  - keyframe_hold pins each reference image across a few frames for stricter adherence.

Outputs one LATENT — the graph is
    loaders → Wan Director → VAEDecode → VHS_VideoCombine.

Reuses wan_processing.py (encode_wan_keyframes) + ltx_director._load_image_tensor.
No audio (Wan has none).
"""
import json
import logging

import torch
import comfy.samplers

from comfy_api.latest import io

from .prompt_relay import get_raw_tokenizer, map_token_indices
from .patches import detect_model_type
from .wan_processing import resolve_wan_dims, encode_wan_keyframes
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


def _plan_clips(prompts, seg_lens, keyframes, total_len, duration_frames, chunk_max):
    """Plan segment-aligned FIRST-LAST-FRAME (FLF) clips — the model both reference
    workflows use. Each keyframe is a hard boundary; the span between consecutive keyframes
    is ONE clip that morphs FROM the keyframe at its start TO the next keyframe (FLF). So a
    clip's last frame is the next clip's first keyframe → the segments truly connect
    (seg1's tail == seg2's head). The span after the final keyframe, and any span longer
    than chunk_max, continue as I2V (chained from the previous clip's last frame).

    Returns clips: ``{"length", "prompt", "start_raw": raw|None, "end_raw": raw|None}``.
      - ``start_raw``: keyframe image to START this clip from — honoured only for clip 0;
        every later clip starts from the PREVIOUS clip's last latent frame (continuity).
      - ``end_raw``: next keyframe image to MORPH TO (FLF); set on the last piece of a span.
    """
    chunk_max = max(5, int(chunk_max))
    dur = max(1, int(duration_frames))

    # Prompt regions over [0, total_len), scaled exactly (no rounding drift).
    glens = _largest_remainder([max(1, l) for l in seg_lens], total_len) if seg_lens else [total_len]
    rprompts = (list(prompts) + [""] * len(glens))[:len(glens)] if prompts else [""] * len(glens)
    regions, c = [], 0
    for p, L in zip(rprompts, glens):
        if L > 0:
            regions.append((c, c + L, p))
            c += L
    if not regions:
        regions = [(0, total_len, "")]

    def prompt_for(a, b):
        best, bestp = 0, regions[0][2]
        for (s, e, p) in regions:
            ov = min(e, b) - max(s, a)
            if ov > best:
                best, bestp = ov, p
        return bestp

    # Keyframe positions in total_len space; a leading non-keyframe span starts image-less.
    kf = sorted((max(0, min(int(round(st * total_len / dur)), total_len - 1)), raw)
                for (st, raw) in keyframes)
    pts = [p for p, _ in kf]
    imgs = [r for _, r in kf]
    if not pts or pts[0] != 0:
        pts = [0] + pts
        imgs = [None] + imgs

    clips = []
    for i in range(len(pts)):
        span_start = pts[i]
        span_end = pts[i + 1] if i + 1 < len(pts) else total_len
        span_len = span_end - span_start
        if span_len <= 0:
            continue
        start_img = imgs[i]
        end_img = imgs[i + 1] if i + 1 < len(pts) else None
        npieces = max(1, (span_len + chunk_max - 1) // chunk_max)
        plens = _largest_remainder([1] * npieces, span_len)
        off = 0
        for pi, plen in enumerate(plens):
            a = span_start + off
            clips.append({
                "length": int(plen),
                "prompt": prompt_for(a, a + plen),
                "start_raw": start_img if pi == 0 else None,
                "end_raw": end_img if pi == len(plens) - 1 else None,
            })
            off += plen
    return clips


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


class WanDirector(io.ComfyNode):
    """Wan timeline editor with segment-aligned first-last-frame (FLF) chaining.
    Drop an image on each timeline segment; each segment becomes an FLF clip that morphs from
    its keyframe to the next segment's keyframe, chained in LATENT space (the previous clip's
    last latent frame seeds the next) — so the segments connect (seg1's tail = seg2's head)
    without colour drift. Outputs one LATENT — feed to VAEDecode → VHS_VideoCombine."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDirector",
            display_name="Wan Director",
            category="WhatDreamsCost",
            description=(
                "Wan2.2/2.1 timeline → long video. Each segment is a first-last-frame (FLF) clip that "
                "morphs from its keyframe to the NEXT segment's keyframe, chained in latent space so "
                "segments connect seamlessly (seg1's tail = seg2's head) without colour drift. Samples "
                "internally (MoE high→low) and outputs one LATENT — feed to VAEDecode → VHS_VideoCombine."
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
                                tooltip="Auto-filled per-segment guide strengths (the timeline image-strength slider, "
                                        "0..1). Each keyframe is enforced with mask = 1 - strength: 1.0 = fully locked "
                                        "to the reference image, lower lets the model deviate from it. Read per-keyframe "
                                        "from the timeline; this widget is just the serialized copy."),
                io.Int.Input("keyframe_hold", default=5, min=1, max=81, step=1, optional=True,
                             tooltip="How many frames to PIN each keyframe (reference image) as known. "
                                     "Higher = stricter adherence to the reference image, at the cost of a "
                                     "brief hold/freeze on that keyframe. Native Wan pins ((hold-1)//4)+1 "
                                     "latent frames: 1 = single frame (loosest), 5 ≈ pin 2 latent frames, "
                                     "9 ≈ 3. On a first-last (FLF) chunk it is capped to half the chunk so "
                                     "there is room to move between the two keyframes."),
                io.Boolean.Input("use_prompt_relay", default=False, optional=True,
                                 tooltip="Reserved (no-op in FLF-clip mode): each clip is conditioned on its own "
                                         "segment prompt directly."),
                io.Float.Input("colormatch_strength", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Reserved (no-op). Clips now chain in LATENT space (no per-clip decode), so "
                                       "there is no decode→re-encode colour drift to correct. Kept for widget-order "
                                       "compatibility."),
                io.Boolean.Input("segment_single_pass", default=False, optional=True,
                                 tooltip="Generate each timeline segment as ONE first-last-frame pass instead of "
                                         "splitting it into chunk_frames pieces. Both ends are anchored (start ≈ this "
                                         "keyframe, end = next keyframe), so mid-segment feature/colour DRIFT is far "
                                         "lower. Cost: a single long pass is much heavier and Wan degrades past ~150 "
                                         "frames, so keep segments short (≈≤150 frames) and/or lower max_side. The "
                                         "last segment has no next keyframe, so it stays I2V (add a final keyframe to "
                                         "anchor its end)."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent",
                                 tooltip="The full chained video latent (all FLF clips stitched in latent space). "
                                         "Feed straight to VAEDecode → VHS_VideoCombine."),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, model_high, clip, vae, global_prompt, global_negative_prompt, width, height,
                length, batch_size, duration_frames, duration_seconds, timeline_data, local_prompts,
                segment_lengths, epsilon=1e-3, steps=8, cfg=1.0, sampler_name="euler",
                scheduler="simple", seed=0, chunk_frames=81, moe_boundary=4,
                i2v_backend="native", frame_rate=16, display_mode="seconds", divisible_by=16,
                max_side=832, guide_strength="", keyframe_hold=5, use_prompt_relay=False,
                colormatch_strength=0.0, segment_single_pass=False, model_low=None, clip_vision=None,
                clip_vision_start=None, clip_vision_end=None) -> io.NodeOutput:

        arch, _patch_size, _stride = detect_model_type(model_high)
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

        # --- Plan segment-aligned FLF clips: seg_i morphs keyframe_i -> keyframe_{i+1}, and
        #     each clip starts from the previous clip's last frame so the segments connect. ---
        # segment_single_pass: one FLF pass per segment (no chunk_frames splitting) so both
        # ends stay anchored — far less mid-segment drift, at the cost of long heavy passes.
        eff_chunk = 100000 if segment_single_pass else chunk_frames
        clips = _plan_clips(prompts, seg_lens, keyframes, total_len, duration_frames, eff_chunk)
        n = len(clips)
        log.info("[WanDirector] %d keyframe(s), %d prompt-seg -> %d FLF clip(s) (single_pass=%s), total~%d frames, %dx%d",
                 len(keyframes), len(prompts), n, segment_single_pass, total_len, tgt_w, tgt_h)

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative_prompt or ""))
        raw_tokenizer = get_raw_tokenizer(clip)
        hold = max(1, int(keyframe_hold))

        stitched = None
        prev_latent_tail = None   # previous clip's last LATENT frame (continuity, no VAE round-trip)
        last_cv = clip_vision_start  # most recent keyframe's CLIP-Vision (carried into continuation)
        for ci, cl in enumerate(clips):
            plen = cl["length"]

            # Continuity is carried in LATENT space (prev_latent_tail) — no decode->re-encode
            # colour drift. Clip 0 starts from its keyframe image; the morph TARGET is the next
            # keyframe (FLF) on a segment's last piece.
            start_image = _load_image_tensor(cl["start_raw"]) if (ci == 0 and cl["start_raw"] is not None) else None
            end_image = _load_image_tensor(cl["end_raw"]) if cl["end_raw"] is not None else None

            # Per-keyframe guide strength (0..1 from the timeline slider; 1 = fully enforced).
            start_str = float(cl["start_raw"].get("guideStrength", 1.0)) if cl["start_raw"] else 1.0
            end_str = float(cl["end_raw"].get("guideStrength", 1.0)) if cl["end_raw"] else 1.0

            inject, cv_outputs = [], []
            if start_image is not None:
                inject.append((0, start_image, hold, start_str))
                cv = clip_vision.encode_image(start_image[:1], crop=True) if clip_vision is not None else clip_vision_start
                if cv is not None:
                    last_cv = cv
                cv_outputs.append(cv)
            if end_image is not None:
                inject.append((max(0, plen - 1), end_image, hold, end_str))
                cv = clip_vision.encode_image(end_image[:1], crop=True) if clip_vision is not None else None
                if cv is not None:
                    last_cv = cv
                cv_outputs.append(cv)
            # No image keyframe this clip (pure continuation): keep the last reference anchor.
            if not cv_outputs and last_cv is not None:
                cv_outputs = [last_cv]

            full_prompt, _tr = map_token_indices(raw_tokenizer, global_prompt, [cl["prompt"]])
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

            positive, neg_c, latent, _lf = encode_wan_keyframes(
                positive, negative, vae, tgt_w, tgt_h, plen, batch_size,
                inject, clip_vision_outputs=cv_outputs,
                latent_context=(prev_latent_tail if ci > 0 else None),
            )

            sampled = _moe_sample(model_high, model_low, int(seed) + ci, int(steps), float(cfg),
                                  sampler_name, scheduler, positive, neg_c, latent, moe_boundary)
            sout = sampled["samples"]

            # Stitch latents (drop the 1-frame overlap each clip shares with the previous tail);
            # decode happens once downstream in VAEDecode.
            stitched = sout if stitched is None else torch.cat([stitched, sout[:, :, 1:]], dim=2)
            if ci < n - 1:
                prev_latent_tail = sout[:, :, -1:]
            log.info("[WanDirector] clip %d/%d: %d frames, flf=%s prompt=%r -> stitched %s",
                     ci + 1, n, plen, end_image is not None, (cl["prompt"] or "")[:24], tuple(stitched.shape))

        return io.NodeOutput({"samples": stitched}, float(frame_rate))
