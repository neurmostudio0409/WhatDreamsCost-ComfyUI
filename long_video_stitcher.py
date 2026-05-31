"""Long Video Stitcher — utilities for chaining multiple Director/sampler chunks
into a single long video latent.

Nodes:

- LongVideoStitcher: concatenates up to 6 latent chunks along the temporal axis.
  Supports three seam modes: `drop` (drop overlap_frames from each preceding
  chunk — Phase 1 behaviour), `linear` (linear crossfade across the overlap),
  `cosine` (smooth cosine crossfade, usually the best quality).

- LatentTailToImage: VAE-decodes the trailing N frames of a latent into a pixel
  image, intended as the start_image for the next chunk's Director timeline.
  This is the "frame_offset" pattern for LTX, which has no native frame_offset
  the way Wan's S2V / Animate / VACE do.

- LightningLoraPreset: applies a Lightning / distilled LoRA and emits the
  recommended sampler config for the chosen preset.

- LongChainSampler: runs KSampler once per chunk (one prompt per line, count
  set by num_chunks), seeding each chunk from the previous chunk's tail and
  stitching the result into one latent.

Audio latents are handled separately — these nodes operate on video latents
only.
"""

import logging
import math

import torch

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes

from comfy_api.latest import io


log = logging.getLogger(__name__)


def _blend_overlap(left_tail, right_head, mode):
    """Crossfade left_tail into right_head along the temporal dim (dim=2).

    Works for 5D video latents [B,C,T,H,W] and 4D audio latents [B,C,T,F] —
    the fade weights broadcast over whatever trailing dims exist. Returns a new
    tensor of the same shape. `mode` is one of "linear" or "cosine".
    """
    T = left_tail.shape[2]
    device = left_tail.device
    dtype = left_tail.dtype
    if mode == "linear":
        alpha = torch.linspace(0.0, 1.0, T, dtype=dtype, device=device)
    else:  # cosine
        t_lin = torch.linspace(0.0, 1.0, T, dtype=dtype, device=device)
        alpha = 0.5 - 0.5 * torch.cos(math.pi * t_lin)
    # Broadcast over batch/channel (dims 0,1) and any trailing dims (H,W or F).
    alpha = alpha.view(*([1, 1, T] + [1] * (left_tail.dim() - 3)))
    return left_tail * (1.0 - alpha) + right_head * alpha


def _concat_latents(latents, overlap_frames, blend_mode="drop", return_seams=False):
    """Stitch latent chunks along the temporal axis (dim=2).

    - blend_mode="drop":   drop the last `overlap_frames` of each preceding chunk
    - blend_mode="linear": crossfade prev tail with next head, linearly
    - blend_mode="cosine": crossfade prev tail with next head, cosine-eased

    For the blend modes, BOTH the trailing N frames of the preceding chunk and
    the leading N frames of the following chunk are consumed by the blend (so
    the seam consumes 2N source frames and emits N blended frames).

    Returns {"samples": stitched_tensor}. Per-chunk noise_mask keys are dropped.
    If return_seams=True, returns (latent_dict, seam_frame_indices) where each
    seam index is the join boundary between two emitted chunk regions in the
    combined tensor (used by SmoothVideoStitcher to locate the bands to re-sample).
    """
    if not latents:
        raise ValueError("LongVideoStitcher: no latents to stitch.")

    samples_list = [lat["samples"] for lat in latents]
    ref = samples_list[0]

    # Audio latents are 4D [B, C, T, F] and have no spatial axes to crossfade — just
    # concatenate them along the temporal axis (dim=2). Each chunk's audio was sampled
    # independently (no seeded overlap the way video chunks share their boundary frame),
    # so there is nothing to drop or blend; the full audio of every chunk is kept.
    if ref.dim() == 4:
        for i, s in enumerate(samples_list[1:], start=1):
            if (s.shape[0], s.shape[1], s.shape[3]) != (ref.shape[0], ref.shape[1], ref.shape[3]):
                raise ValueError(
                    f"LongVideoStitcher: audio chunk {i + 1} shape {tuple(s.shape)} does not "
                    f"match chunk 1 shape {tuple(ref.shape)} (batch / channels / freq-bins must agree)."
                )
        combined = torch.cat(samples_list, dim=2)
        log.info(
            "[LongVideoStitcher] Concatenated %d audio latents -> %s",
            len(latents), tuple(combined.shape),
        )
        out = {"samples": combined, "type": latents[0].get("type", "audio")}
        if return_seams:
            seams, c = [], 0
            for s in samples_list[:-1]:
                c += s.shape[2]
                seams.append(c)
            return out, seams
        return out

    # Sanity-check shapes match on B / C / H / W.
    for i, s in enumerate(samples_list[1:], start=1):
        if (s.shape[0], s.shape[1], s.shape[3], s.shape[4]) != \
           (ref.shape[0], ref.shape[1], ref.shape[3], ref.shape[4]):
            raise ValueError(
                f"LongVideoStitcher: chunk {i + 1} shape {tuple(s.shape)} does not "
                f"match chunk 1 shape {tuple(ref.shape)} (batch / channels / spatial "
                "dims must agree)."
            )

    if overlap_frames <= 0 or blend_mode == "drop":
        # Drop mode: trim trailing overlap_frames from every chunk except the last.
        parts = []
        for i, s in enumerate(samples_list):
            if overlap_frames > 0 and i < len(samples_list) - 1 and s.shape[2] > overlap_frames:
                s = s[:, :, :-overlap_frames]
            parts.append(s)
        combined = torch.cat(parts, dim=2)
    else:
        # Crossfade mode: blend overlap region across each seam.
        parts = [samples_list[0]]
        for i in range(1, len(samples_list)):
            prev = parts[-1]
            curr = samples_list[i]
            # If either side is shorter than overlap_frames, fall back to drop
            # to avoid mangling short chunks.
            if prev.shape[2] <= overlap_frames or curr.shape[2] <= overlap_frames:
                if prev.shape[2] > overlap_frames:
                    parts[-1] = prev[:, :, :-overlap_frames]
                parts.append(curr)
                continue

            prev_main = prev[:, :, :-overlap_frames]
            prev_tail = prev[:, :, -overlap_frames:]
            curr_head = curr[:, :, :overlap_frames]
            curr_main = curr[:, :, overlap_frames:]
            blended = _blend_overlap(prev_tail, curr_head, blend_mode)
            parts[-1] = torch.cat([prev_main, blended], dim=2)
            parts.append(curr_main)
        combined = torch.cat(parts, dim=2)

    log.info(
        "[LongVideoStitcher] Stitched %d chunks with mode=%s overlap=%d -> %s",
        len(latents), blend_mode, overlap_frames, tuple(combined.shape),
    )
    out = {"samples": combined}
    if return_seams:
        # Join boundaries = cumulative lengths of all emitted parts except the last.
        seams, c = [], 0
        for p in parts[:-1]:
            c += p.shape[2]
            seams.append(c)
        return out, seams
    return out


_BLEND_MODES = ["drop", "linear", "cosine"]


class LongVideoStitcher(io.ComfyNode):
    """Concatenate up to 6 video-latent chunks into a single long latent.

    Typical use: run the Director + Sampler N times (each chunk seeded from the
    previous chunk's last frame via `LatentTailToImage`), connect the resulting
    latents in order, and feed the stitched output into a single VAE Decode.

    For the smoothest seam between chunks use `blend_mode="cosine"` with
    `overlap_frames` set to the number of latent frames you want the crossfade
    to span — both sides of the seam contribute, so a small overlap (2–4
    latent frames) is usually enough to hide the boundary.
    """

    # Max number of latent input slots. The JS extension (js/long_video_stitcher.js)
    # grows/shrinks the *visible* slots dynamically as you connect chunks; the backend
    # just needs every possible slot declared, so connect as many as you need up to this.
    MAX_LATENTS = 12

    @classmethod
    def define_schema(cls):
        latent_inputs = [io.Latent.Input("latent_1", tooltip="First chunk (required).")]
        for i in range(2, cls.MAX_LATENTS + 1):
            latent_inputs.append(
                io.Latent.Input(f"latent_{i}", optional=True, tooltip=f"Chunk {i} (optional).")
            )
        return io.Schema(
            node_id="LongVideoStitcher",
            display_name="Long Video Stitcher",
            category="WhatDreamsCost",
            description=(
                f"Concatenates up to {cls.MAX_LATENTS} video latents along the temporal axis "
                "(connect as many as you need — the input slots grow dynamically). "
                "blend_mode=drop trims the trailing overlap_frames of each preceding "
                "chunk; blend_mode=linear/cosine crossfades the overlap region across "
                "each seam for a smoother boundary. Channels / spatial dims of all "
                "chunks must match. Also accepts 4D audio latents ([B,C,T,F]) — those "
                "are plainly concatenated (overlap_frames / blend_mode are ignored) so "
                "you can stitch the matching audio of chained chunks with a second instance."
            ),
            inputs=[
                *latent_inputs,
                io.Int.Input(
                    "overlap_frames", default=0, min=0, max=64, step=1, optional=True,
                    tooltip=(
                        "Latent frames to consume at each seam. For drop mode this is the "
                        "number of frames trimmed from each preceding chunk's tail. For "
                        "linear/cosine this is the crossfade width — both sides of the seam "
                        "contribute, so 2–4 is usually enough."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode", options=_BLEND_MODES, default="drop", optional=True,
                    tooltip=(
                        "drop = trim overlap from preceding chunk (Phase 1 behaviour). "
                        "linear = linear crossfade across overlap. cosine = smooth cosine "
                        "crossfade (best quality for visible seams). drop is the safest "
                        "choice when chunk K+1 was seeded from chunk K's tail and you don't "
                        "want the blend re-introducing the duplicate."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent_1, latent_2=None, latent_3=None, latent_4=None,
                latent_5=None, latent_6=None, latent_7=None, latent_8=None,
                latent_9=None, latent_10=None, latent_11=None, latent_12=None,
                overlap_frames=0, blend_mode="drop") -> io.NodeOutput:
        # Collect latent_1..latent_12 in order, keeping only the connected ones.
        ordered = [latent_1, latent_2, latent_3, latent_4, latent_5, latent_6,
                   latent_7, latent_8, latent_9, latent_10, latent_11, latent_12]
        chunks = [l for l in ordered if l is not None]
        out = _concat_latents(chunks, int(overlap_frames), blend_mode)
        return io.NodeOutput(out)


class SmoothVideoStitcher(io.ComfyNode):
    """Join video-latent chunks with a smooth seam — crossfade and/or MODEL re-gen.

    Used like LongVideoStitcher: connect chunk latents in order (latent_1..N grow
    dynamically) and pick `overlap_frames` + `blend_mode` (drop / linear / cosine)
    for the base seam blend.

    If you ALSO connect `model` + `clip`, the node then re-samples a narrow band of
    `transition_frames` latent frames straddling every seam with the model, so the
    boundary becomes a *generated* transition rather than just a crossfade. Chunk
    interiors are locked (noise_mask = 0); only the seam bands are denoised. No
    prompt is needed — a blank conditioning is built from `clip` internally.

    If model/clip are left unconnected the node is a plain crossfade stitcher.
    Use a PLAIN model (NOT one patched with a prompt-relay mask for a single
    chunk's shape). Video latents only (5D); for audio use Smooth Audio Stitcher.
    """

    MAX_LATENTS = 12

    @classmethod
    def define_schema(cls):
        latent_inputs = [io.Latent.Input("latent_1", tooltip="First chunk (required).")]
        for i in range(2, cls.MAX_LATENTS + 1):
            latent_inputs.append(
                io.Latent.Input(f"latent_{i}", optional=True, tooltip=f"Chunk {i} (optional).")
            )
        return io.Schema(
            node_id="SmoothVideoStitcher",
            display_name="Smooth Video Stitcher",
            category="WhatDreamsCost",
            description=(
                "Stitches video-latent chunks with a crossfade seam (overlap_frames + "
                "blend_mode), and — if a model + clip are connected — re-samples a small "
                "band at each seam with the model so the boundary is a generated transition "
                "instead of a cut/dissolve. No prompt needed (blank conditioning from clip). "
                "Connect chunks to latent_1..N (slots grow). Video latents only."
            ),
            inputs=[
                *latent_inputs,
                io.Int.Input(
                    "overlap_frames", default=0, min=0, max=64, step=1, optional=True,
                    tooltip="Latent frames consumed at each seam. drop = trimmed from each preceding chunk; linear/cosine = crossfade width (both sides contribute).",
                ),
                io.Combo.Input(
                    "blend_mode", options=_BLEND_MODES, default="cosine", optional=True,
                    tooltip="drop = hard cut (trim overlap). linear/cosine = crossfade the overlap. cosine is the smoothest base blend.",
                ),
                io.Model.Input("model", optional=True, tooltip="Optional. Connect to re-sample seam bands with the model (a generated transition). Use a PLAIN model, not one patched with a prompt-relay mask. Leave empty for crossfade-only."),
                io.Clip.Input("clip", optional=True, tooltip="Optional. Required only when model is connected — used to build a blank (no-prompt) conditioning for the seam re-sampling."),
                io.Int.Input(
                    "transition_frames", default=2, min=1, max=16, step=1, optional=True,
                    tooltip="(model re-gen) Latent frames regenerated on EACH side of every seam. Larger = longer transition.",
                ),
                io.Float.Input(
                    "denoise", default=0.7, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="(model re-gen) How hard the seam band is regenerated. Higher = stronger bridge; lower = preserve more of the crossfaded frames.",
                ),
                io.Int.Input("steps", default=20, min=1, max=200, step=1, optional=True),
                io.Float.Input("cfg", default=3.0, min=0.0, max=20.0, step=0.1, optional=True),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", optional=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal", optional=True),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff, optional=True,
                    tooltip="Seed for the seam re-sampling.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent_1, latent_2=None, latent_3=None, latent_4=None,
                latent_5=None, latent_6=None, latent_7=None, latent_8=None,
                latent_9=None, latent_10=None, latent_11=None, latent_12=None,
                overlap_frames=0, blend_mode="cosine", model=None, clip=None,
                transition_frames=2, denoise=0.7, steps=20, cfg=3.0,
                sampler_name="euler", scheduler="normal", seed=0) -> io.NodeOutput:
        ordered = [latent_1, latent_2, latent_3, latent_4, latent_5, latent_6,
                   latent_7, latent_8, latent_9, latent_10, latent_11, latent_12]
        chunks = [l for l in ordered if l is not None]
        if not chunks:
            raise ValueError("SmoothVideoStitcher: no latents to stitch.")
        if chunks[0]["samples"].dim() != 5:
            raise ValueError(
                f"SmoothVideoStitcher: expected 5D video latents [B,C,T,H,W], got shape "
                f"{tuple(chunks[0]['samples'].shape)}. (For audio use Smooth Audio Stitcher.)"
            )

        # Base stitch (crossfade / drop) + the seam join positions.
        base, seams = _concat_latents(chunks, int(overlap_frames), blend_mode, return_seams=True)

        # Crossfade-only when no model/clip, or nothing to bridge.
        if model is None or clip is None or not seams:
            return io.NodeOutput(base)

        combined = base["samples"]
        total_t = combined.shape[2]
        tf = max(1, int(transition_frames))
        ctx = tf  # preserved context frames each side of the band, fed to the model

        # Blank (no-prompt) conditioning so the seam is just smoothed, not steered.
        empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        log.info(
            "[SmoothVideoStitcher] %d chunks -> %d latent frames, %d seam(s) at %s, "
            "regen band=%d each side (+%d ctx), denoise=%.2f — windowed re-sample",
            len(chunks), total_t, len(seams), seams, tf, ctx, float(denoise),
        )

        # Re-sample ONLY a small window around each seam, not the whole video. Each
        # window = the regen band [seam-tf, seam+tf] plus `ctx` preserved context
        # frames on each side (so the model has temporal context). We then paste just
        # the regenerated band back. Sampling cost is O(band) per seam instead of
        # O(total video length) — the slow part of the old whole-latent pass.
        out = combined.clone()
        for k, seam in enumerate(seams):
            lo = max(0, seam - tf - ctx)
            hi = min(total_t, seam + tf + ctx)
            band_lo = max(0, (seam - tf) - lo)
            band_hi = min(hi - lo, (seam + tf) - lo)
            if band_hi <= band_lo:
                continue
            win = out[:, :, lo:hi].clone()
            wlen = win.shape[2]
            wmask = torch.zeros((win.shape[0], 1, wlen, 1, 1), dtype=win.dtype, device=win.device)
            wmask[:, :, band_lo:band_hi] = 1.0
            sampled = nodes.common_ksampler(
                model, (int(seed) + k) & 0xffffffffffffffff, int(steps), float(cfg),
                sampler_name, scheduler, empty, empty,
                {"samples": win, "noise_mask": wmask}, denoise=float(denoise),
            )[0]["samples"]
            out[:, :, lo + band_lo: lo + band_hi] = sampled[:, :, band_lo:band_hi]

        return io.NodeOutput({"samples": out})


def _concat_audio_latents(latents, overlap_frames, blend_mode="cosine"):
    """Join 4D audio latents [B,C,T,F] along the temporal axis with a crossfade.

    blend_mode "cosine"/"linear" crossfades the trailing `overlap_frames` of each
    preceding chunk with the leading `overlap_frames` of the next (both sides are
    consumed, one blended span is emitted) so the seam fades smoothly. "drop" trims
    the trailing `overlap_frames` from each preceding chunk (a hard join with no
    fade). "concat" just butts the chunks together (no trim, no fade).
    """
    if not latents:
        raise ValueError("SmoothAudioStitcher: no latents to stitch.")

    samples_list = [l["samples"] for l in latents]
    ref = samples_list[0]
    if ref.dim() != 4:
        raise ValueError(
            f"SmoothAudioStitcher: expected 4D audio latents [B,C,T,F], got shape "
            f"{tuple(ref.shape)}. (For video latents use Smooth Video Stitcher.)"
        )
    for i, s in enumerate(samples_list[1:], start=1):
        if (s.shape[0], s.shape[1], s.shape[3]) != (ref.shape[0], ref.shape[1], ref.shape[3]):
            raise ValueError(
                f"SmoothAudioStitcher: chunk {i + 1} shape {tuple(s.shape)} does not match "
                f"chunk 1 {tuple(ref.shape)} (batch / channels / freq-bins must agree)."
            )

    out_type = latents[0].get("type", "audio")
    if len(samples_list) == 1:
        return {"samples": ref, "type": out_type}

    if overlap_frames <= 0 or blend_mode == "concat":
        combined = torch.cat(samples_list, dim=2)
    elif blend_mode == "drop":
        # Trim trailing overlap_frames from every chunk except the last (hard join).
        parts = []
        for i, s in enumerate(samples_list):
            if i < len(samples_list) - 1 and s.shape[2] > overlap_frames:
                s = s[:, :, :-overlap_frames]
            parts.append(s)
        combined = torch.cat(parts, dim=2)
    else:
        parts = [samples_list[0]]
        for i in range(1, len(samples_list)):
            prev = parts[-1]
            curr = samples_list[i]
            # If either side is shorter than the overlap, fall back to plain concat
            # for this seam to avoid mangling a short chunk.
            if prev.shape[2] <= overlap_frames or curr.shape[2] <= overlap_frames:
                parts.append(curr)
                continue
            prev_main = prev[:, :, :-overlap_frames]
            prev_tail = prev[:, :, -overlap_frames:]
            curr_head = curr[:, :, :overlap_frames]
            curr_main = curr[:, :, overlap_frames:]
            blended = _blend_overlap(prev_tail, curr_head, blend_mode)
            parts[-1] = torch.cat([prev_main, blended], dim=2)
            parts.append(curr_main)
        combined = torch.cat(parts, dim=2)

    log.info(
        "[SmoothAudioStitcher] Stitched %d audio latents with mode=%s overlap=%d -> %s",
        len(latents), blend_mode, overlap_frames, tuple(combined.shape),
    )
    return {"samples": combined, "type": out_type}


_AUDIO_BLEND_MODES = ["cosine", "linear", "drop", "concat"]


class SmoothAudioStitcher(io.ComfyNode):
    """Join AUDIO-latent chunks with a smooth crossfade at each seam.

    The audio counterpart to Smooth Video Stitcher. Connect your chunks' audio
    latents (4D [B,C,T,F]) to latent_1..N (slots grow as you wire them) and the
    node crossfades the overlap at every seam so the audio transitions smoothly
    instead of cutting. Pure DSP — no model needed (an audio crossfade already
    sounds smooth, and LTX audio is sampled jointly with video so there is no
    standalone audio model to re-generate a seam with).

    Audio latents only (4D); for video use Smooth Video Stitcher.
    """

    MAX_LATENTS = 12

    @classmethod
    def define_schema(cls):
        latent_inputs = [io.Latent.Input("latent_1", tooltip="First chunk's audio latent (required).")]
        for i in range(2, cls.MAX_LATENTS + 1):
            latent_inputs.append(
                io.Latent.Input(f"latent_{i}", optional=True, tooltip=f"Chunk {i} audio latent (optional).")
            )
        return io.Schema(
            node_id="SmoothAudioStitcher",
            display_name="Smooth Audio Stitcher",
            category="WhatDreamsCost",
            description=(
                "Concatenates audio latents and crossfades the overlap at each seam "
                "(cosine/linear) so chained chunks' audio transitions smoothly. Connect "
                "chunks to latent_1..N (slots grow as you wire them). Pure crossfade — no "
                "model needed. Audio latents only (4D); for video use Smooth Video Stitcher."
            ),
            inputs=[
                *latent_inputs,
                io.Int.Input(
                    "overlap_frames", default=4, min=0, max=64, step=1, optional=True,
                    tooltip="Audio latent frames crossfaded at each seam (both sides contribute). 0 / concat = no fade (plain concatenate).",
                ),
                io.Combo.Input(
                    "blend_mode", options=_AUDIO_BLEND_MODES, default="cosine", optional=True,
                    tooltip="cosine = smooth eased crossfade (best). linear = constant-rate crossfade. drop = trim overlap from each preceding chunk (hard join, no fade). concat = just butt the chunks together (no trim, no fade).",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent_1, latent_2=None, latent_3=None, latent_4=None,
                latent_5=None, latent_6=None, latent_7=None, latent_8=None,
                latent_9=None, latent_10=None, latent_11=None, latent_12=None,
                overlap_frames=4, blend_mode="cosine") -> io.NodeOutput:
        ordered = [latent_1, latent_2, latent_3, latent_4, latent_5, latent_6,
                   latent_7, latent_8, latent_9, latent_10, latent_11, latent_12]
        chunks = [l for l in ordered if l is not None]
        out = _concat_audio_latents(chunks, int(overlap_frames), blend_mode)
        return io.NodeOutput(out)


class SmoothAudioJoin(io.ComfyNode):
    """Join per-chunk AUDIO with a smooth waveform crossfade at each seam.

    Takes already-decoded AUDIO (one per chunk — decode each chunk's audio latent with
    LTXVAudioVAEDecode first), crossfades the WAVEFORMS at every seam (cosine/linear),
    and concatenates. Because it works in the waveform domain — not the audio latent —
    it cannot produce an undecodable latent (the LTX audio VAE is causal and picky about
    latent length), and an audio crossfade sounds smooth. Output is one AUDIO you can
    feed straight into CreateVideo.

    This is the safe way to smooth the audio seam when chaining chunks; for video use
    Smooth Video Stitcher / LTX Smooth Transition.
    """

    MAX_AUDIO = 12

    @classmethod
    def define_schema(cls):
        audio_inputs = [io.Audio.Input("audio_1", tooltip="First chunk's decoded AUDIO (required).")]
        for i in range(2, cls.MAX_AUDIO + 1):
            audio_inputs.append(
                io.Audio.Input(f"audio_{i}", optional=True, tooltip=f"Chunk {i} decoded AUDIO (optional).")
            )
        return io.Schema(
            node_id="SmoothAudioJoin",
            display_name="Smooth Audio Join",
            category="WhatDreamsCost",
            description=(
                "Crossfades per-chunk AUDIO waveforms at each seam and concatenates them into "
                "one AUDIO. Decode each chunk's audio latent (e.g. LTXVAudioVAEDecode) first and "
                "connect the AUDIO outputs to audio_1..N (slots grow). Works in the waveform "
                "domain so it never breaks the audio VAE. Feed the output into CreateVideo."
            ),
            inputs=[
                *audio_inputs,
                io.Float.Input(
                    "crossfade_seconds", default=0.25, min=0.0, max=5.0, step=0.01, optional=True,
                    tooltip="Length of the crossfade at each seam, in seconds. 0 = hard concatenate (no fade).",
                ),
                io.Combo.Input(
                    "blend_mode", options=["cosine", "linear"], default="cosine", optional=True,
                    tooltip="Crossfade curve. cosine = smooth eased fade (best). linear = constant-rate fade.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, audio_1, audio_2=None, audio_3=None, audio_4=None, audio_5=None, audio_6=None,
                audio_7=None, audio_8=None, audio_9=None, audio_10=None, audio_11=None, audio_12=None,
                crossfade_seconds=0.25, blend_mode="cosine") -> io.NodeOutput:
        ordered = [audio_1, audio_2, audio_3, audio_4, audio_5, audio_6,
                   audio_7, audio_8, audio_9, audio_10, audio_11, audio_12]
        auds = [a for a in ordered if a is not None]
        if not auds:
            raise ValueError("SmoothAudioJoin: no audio connected.")

        sr = int(auds[0]["sample_rate"])
        wavs = []
        for i, a in enumerate(auds):
            if int(a["sample_rate"]) != sr:
                raise ValueError(
                    f"SmoothAudioJoin: clip {i + 1} sample_rate {a['sample_rate']} != clip 1 {sr}."
                )
            w = a["waveform"]
            if w.dim() == 2:       # [C, S] -> [1, C, S]
                w = w.unsqueeze(0)
            wavs.append(w)

        ref = wavs[0]
        for i, w in enumerate(wavs[1:], start=1):
            if w.shape[0] != ref.shape[0] or w.shape[1] != ref.shape[1]:
                raise ValueError(
                    f"SmoothAudioJoin: clip {i + 1} waveform {tuple(w.shape)} batch/channels do "
                    f"not match clip 1 {tuple(ref.shape)}."
                )

        n = max(0, int(round(float(crossfade_seconds) * sr)))
        if len(wavs) == 1:
            combined = wavs[0]
        elif n == 0:
            combined = torch.cat(wavs, dim=2)
        else:
            parts = [wavs[0]]
            for i in range(1, len(wavs)):
                prev = parts[-1]
                curr = wavs[i]
                if prev.shape[2] <= n or curr.shape[2] <= n:
                    parts.append(curr)  # too short to fade — just butt together
                    continue
                prev_main = prev[:, :, :-n]
                prev_tail = prev[:, :, -n:]
                curr_head = curr[:, :, :n]
                curr_main = curr[:, :, n:]
                blended = _blend_overlap(prev_tail, curr_head, blend_mode)  # broadcasts over [B,C,·]
                parts[-1] = torch.cat([prev_main, blended], dim=2)
                parts.append(curr_main)
            combined = torch.cat(parts, dim=2)

        log.info(
            "[SmoothAudioJoin] joined %d clip(s) @ %dHz, crossfade=%.2fs (%d samples) -> %s",
            len(wavs), sr, float(crossfade_seconds), n, tuple(combined.shape),
        )
        return io.NodeOutput({"waveform": combined, "sample_rate": sr})


class LatentTailToImage(io.ComfyNode):
    """VAE-decode the trailing N frames of a video latent into a pixel image.

    Use this to chain LTX chunks: feed the previous chunk's latent in, get the
    last frame(s) as an IMAGE, then drop that image onto the next LTX
    Director's timeline as the start keyframe of chunk K+1.

    For Wan, the WanS2VDirector / WanAnimateDirector / WanVaceDirector already
    expose `frame_offset` outputs designed for chaining — use those instead.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentTailToImage",
            display_name="Latent Tail to Image",
            category="WhatDreamsCost",
            description=(
                "Decodes the last N latent frames of a video latent back into a pixel "
                "IMAGE tensor. Intended as the bridge between chunked LTX generations: "
                "the tail of chunk K becomes the start keyframe of chunk K+1."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Video latent (output of a sampler)."),
                io.Vae.Input("vae", tooltip="VAE matching the latent."),
                io.Int.Input(
                    "num_latent_frames", default=1, min=1, max=64, step=1,
                    tooltip=(
                        "How many trailing LATENT frames to decode. 1 = a single image "
                        "(typical). >1 gives a short video chunk that can seed a multi-"
                        "frame keyframe / continue_motion input."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, num_latent_frames=1) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.dim() != 5:
            raise ValueError(
                f"LatentTailToImage: expected a 5D video latent [B,C,T,H,W], got shape "
                f"{tuple(samples.shape)}."
            )
        n = max(1, min(int(num_latent_frames), samples.shape[2]))
        tail = samples[:, :, -n:].contiguous()
        decoded = vae.decode(tail)
        # ComfyUI VAE.decode for video returns [T, H, W, C] (4D). If it returns 5D
        # squeeze the batch dim.
        if decoded.dim() == 5:
            decoded = decoded[0]
        log.info(
            "[LatentTailToImage] Decoded last %d latent frame(s): %s -> image %s",
            n, tuple(samples.shape), tuple(decoded.shape),
        )
        return io.NodeOutput(decoded)


# ---------------------------------------------------------------------------
# Lightning LoRA preset
# ---------------------------------------------------------------------------


# Preset name -> (steps, cfg, sampler_name, scheduler).
# Tuned for the most common community Lightning distillations. "custom" leaves
# the sampler params at sensible defaults but the user is expected to override.
_LIGHTNING_PRESETS = {
    "LTX 4-step":  (4, 1.0, "euler",   "simple"),
    "LTX 8-step":  (8, 1.0, "euler",   "simple"),
    "Wan 4-step":  (4, 1.0, "euler",   "simple"),
    "Wan 6-step":  (6, 1.0, "euler",   "simple"),
    "Wan 8-step":  (8, 1.0, "euler",   "simple"),
    "custom":      (8, 1.0, "euler",   "normal"),
}


class LightningLoraPreset(io.ComfyNode):
    """Apply a Lightning / distilled LoRA and emit the recommended sampler config.

    Bundles `LoraLoader` + the "what steps / cfg / sampler / scheduler do I
    use for this 4-step Lightning thing?" lookup into one node. Wire the
    `steps / cfg / sampler_name / scheduler` outputs into KSampler (or
    LongChainSampler) so you don't have to remember the magic numbers.

    For Wan2.2 MoE 14B, this only patches one model. If you want the
    high-noise and low-noise pair both patched, chain two of these nodes (one
    per model) with the same LoRA, or call LoraLoaderModelOnly twice.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LightningLoraPreset",
            display_name="Lightning LoRA Preset",
            category="WhatDreamsCost",
            description=(
                "Apply a Lightning / distilled LoRA to the model (and optionally CLIP) "
                "and output the recommended sampler config for the chosen preset. "
                "Wire steps/cfg/sampler_name/scheduler into KSampler or LongChainSampler."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Diffusion model to apply the LoRA to."),
                io.Clip.Input("clip", optional=True, tooltip="Optional CLIP to also patch."),
                io.Combo.Input(
                    "lora_name", options=folder_paths.get_filename_list("loras"),
                    tooltip="The Lightning / distilled LoRA to apply.",
                ),
                io.Float.Input(
                    "strength_model", default=1.0, min=-10.0, max=10.0, step=0.05,
                    tooltip="LoRA strength on the diffusion model. 1.0 = full effect.",
                ),
                io.Float.Input(
                    "strength_clip", default=1.0, min=-10.0, max=10.0, step=0.05, optional=True,
                    tooltip="LoRA strength on CLIP (ignored if no clip is connected).",
                ),
                io.Combo.Input(
                    "preset", options=list(_LIGHTNING_PRESETS.keys()), default="LTX 4-step",
                    tooltip=(
                        "Sampler recipe. Affects the steps/cfg/sampler_name/scheduler outputs "
                        "only — the LoRA is always applied. 'custom' = pass-through defaults; "
                        "override the sampler params manually."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Int.Output(display_name="steps"),
                io.Float.Output(display_name="cfg"),
                io.Combo.Output(
                    display_name="sampler_name",
                    options=list(comfy.samplers.KSampler.SAMPLERS),
                ),
                io.Combo.Output(
                    display_name="scheduler",
                    options=list(comfy.samplers.KSampler.SCHEDULERS),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, preset,
                clip=None, strength_clip=1.0) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # load_lora_for_models accepts clip=None; the second return is then None too.
        # We swap a None clip back to whatever the caller passed (often None) so the
        # node's CLIP output passes through cleanly.
        model_out, clip_out = comfy.sd.load_lora_for_models(
            model, clip, lora, float(strength_model), float(strength_clip),
        )
        if clip_out is None:
            clip_out = clip

        steps, cfg, sampler_name, scheduler = _LIGHTNING_PRESETS[preset]
        log.info(
            "[LightningLoraPreset] Applied %s @ model=%.2f clip=%.2f, preset=%s "
            "(steps=%d, cfg=%.2f, sampler=%s, scheduler=%s)",
            lora_name, strength_model, strength_clip, preset,
            steps, cfg, sampler_name, scheduler,
        )
        return io.NodeOutput(model_out, clip_out, steps, cfg, sampler_name, scheduler)


# ---------------------------------------------------------------------------
# Dynamic N-chunk sampler — one prompt per chunk via a multiline list
# ---------------------------------------------------------------------------


def _sample_and_stitch_chunks(model, positives, negative, base_latent, overlap,
                              steps, cfg, sampler_name, scheduler, seed, denoise):
    """Sample one chunk per conditioning in `positives`, seeding each chunk after
    the first from the previous chunk's tail (front `overlap` latent frames locked
    via noise_mask=0), then stitch by dropping each chunk's seeded prefix.

    Used by LongChainSampler. Returns {"samples": stitched}.
    """
    base_samples = base_latent["samples"]
    if base_samples.dim() != 5:
        raise ValueError(
            f"LongChainSampler: expected a 5D video latent [B,C,T,H,W], got shape "
            f"{tuple(base_samples.shape)}."
        )
    base_T = base_samples.shape[2]
    overlap = max(1, int(overlap))
    if overlap >= base_T:
        raise ValueError(
            f"LongChainSampler: seed_overlap_latent_frames ({overlap}) must be smaller than "
            f"the latent's temporal length ({base_T})."
        )

    all_chunks = []
    for i, positive in enumerate(positives):
        chunk_seed = (int(seed) + i) & 0xffffffffffffffff
        if i == 0:
            chunk_latent = base_latent
        else:
            prev = all_chunks[-1]["samples"]
            new_samples = base_samples.clone()
            copy = min(overlap, prev.shape[2])
            new_samples[:, :, :copy] = prev[:, :, -copy:]
            mask = torch.ones(
                (new_samples.shape[0], 1, new_samples.shape[2], 1, 1),
                dtype=new_samples.dtype, device=new_samples.device,
            )
            mask[:, :, :copy] = 0.0
            chunk_latent = {"samples": new_samples, "noise_mask": mask}

        log.info("[LongChainSampler] Chunk %d/%d: seed=%d", i + 1, len(positives), chunk_seed)
        sampled = nodes.common_ksampler(
            model, chunk_seed, int(steps), float(cfg), sampler_name, scheduler,
            positive, negative, chunk_latent, denoise=float(denoise),
        )[0]
        all_chunks.append(sampled)

    parts = [all_chunks[0]["samples"]]
    for chunk in all_chunks[1:]:
        s = chunk["samples"]
        parts.append(s[:, :, overlap:] if overlap < s.shape[2] else s)
    final = torch.cat(parts, dim=2)
    log.info("[LongChainSampler] Final stitched latent: %s (chunks=%d, per-chunk T=%d, overlap=%d)",
             tuple(final.shape), len(positives), base_T, overlap)
    return {"samples": final}


class LongChainSampler(io.ComfyNode):
    """Dynamic long-video sampler — set the chunk count and feed one prompt per chunk.

    The per-chunk prompts are a multiline list — one prompt per line (`|` also
    separates) — so the chunk count is unbounded. `num_chunks` drives how many
    chunks are generated:

    - num_chunks = 0  → generate exactly one chunk per prompt line.
    - num_chunks = N  → generate N chunks; if there are fewer prompt lines than N the
      last line is held for the remaining chunks (so 1 line + N=10 = the same prompt
      held across 10 chunks; 3 lines + N=10 = prompts 1,2,3 then 3 held).

    Each chunk is seeded from the previous chunk's tail (front
    `seed_overlap_latent_frames` locked via noise_mask) and the chunks are stitched
    into one latent. This is the fast single-stage path; for the LTX 2-stage upscale
    + guide quality, chain LTX Directors with prev_latent instead.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongChainSampler",
            display_name="Long Chain Sampler (Dynamic)",
            category="WhatDreamsCost",
            description=(
                "Runs KSampler once per chunk, seeding each chunk from the previous chunk's "
                "tail, and stitches the result into one long latent. Per-chunk prompts are a "
                "multiline list (one per line); num_chunks sets the count (0 = one per line, "
                "N = hold the last line for any extra chunks). Not capped by input sockets."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Diffusion model (a plain model — not one already patched with a prompt-relay mask for a different latent shape)."),
                io.Clip.Input("clip", tooltip="CLIP used to encode each chunk's prompt."),
                io.Latent.Input("latent", tooltip="Chunk-length latent template. Its temporal length is the per-chunk length; total output ≈ T × num_chunks − (num_chunks−1) × seed_overlap_latent_frames."),
                io.String.Input(
                    "prompts", multiline=True, default="",
                    tooltip="Per-chunk prompts, ONE PER LINE (a '|' also splits). Chunk 1 uses line 1, chunk 2 line 2, etc.",
                ),
                io.String.Input(
                    "global_prompt", multiline=True, default="", optional=True,
                    tooltip="Optional text prepended to every chunk's prompt — anchors persistent subject / style across the whole video.",
                ),
                io.Conditioning.Input("negative", optional=True, tooltip="Optional negative conditioning. An empty one is built from clip if unconnected."),
                io.Int.Input(
                    "num_chunks", default=0, min=0, max=1000, step=1,
                    tooltip="How many chunks to generate. 0 = one chunk per prompt line. If N exceeds the prompt-line count, the last line is held for the remaining chunks.",
                ),
                io.Int.Input(
                    "seed_overlap_latent_frames", default=1, min=1, max=16, step=1,
                    tooltip="Latent frames at the front of each chunk (after the first) locked to the previous chunk's tail. Drives continuity at every seam.",
                ),
                io.Int.Input("steps", default=20, min=1, max=200, step=1),
                io.Float.Input("cfg", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    tooltip="Base seed. Chunk K uses seed + K so chunks don't collapse to identical noise.",
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, prompts, global_prompt="", negative=None,
                num_chunks=0, seed_overlap_latent_frames=1, steps=20, cfg=3.0,
                sampler_name="euler", scheduler="normal", seed=0, denoise=1.0) -> io.NodeOutput:
        # Parse the multiline prompt list ('|' also separates), dropping blank lines.
        lines = [seg.strip() for raw in prompts.split("\n") for seg in raw.split("|") if seg.strip()]
        if not lines:
            raise ValueError("LongChainSampler: 'prompts' is empty — provide at least one prompt line.")

        n = int(num_chunks) if int(num_chunks) > 0 else len(lines)
        gp = global_prompt.strip()

        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        positives = []
        for i in range(n):
            line = lines[min(i, len(lines) - 1)]  # hold the last line for extra chunks
            full = f"{gp} {line}".strip() if gp else line
            positives.append(clip.encode_from_tokens_scheduled(clip.tokenize(full)))

        log.info("[LongChainSampler] %d chunk(s) from %d prompt line(s)%s",
                 n, len(lines), " (+global)" if gp else "")

        out = _sample_and_stitch_chunks(
            model, positives, negative, latent, seed_overlap_latent_frames,
            steps, cfg, sampler_name, scheduler, seed, denoise,
        )
        return io.NodeOutput(out)
