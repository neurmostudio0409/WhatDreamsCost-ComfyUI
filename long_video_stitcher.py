"""Long Video Stitcher — utilities for chaining multiple Director/sampler chunks
into a single long video latent.

Three nodes:

- LongVideoStitcher: concatenates up to 6 latent chunks along the temporal axis.
  Supports three seam modes: `drop` (drop overlap_frames from each preceding
  chunk — Phase 1 behaviour), `linear` (linear crossfade across the overlap),
  `cosine` (smooth cosine crossfade, usually the best quality).

- LatentTailToImage: VAE-decodes the trailing N frames of a latent into a pixel
  image, intended as the start_image for the next chunk's Director timeline.
  This is the "frame_offset" pattern for LTX, which has no native frame_offset
  the way Wan's S2V / Animate / VACE do.

- LongChunkSampler: runs KSampler N times in one node, seeding each chunk from
  the previous chunk's tail latent (the front `seed_overlap_latent_frames` of
  every chunk after the first is locked to the previous chunk's tail via a
  noise_mask). Output is a single stitched latent. Best for "hold B for a long
  time" workflows where the conditioning is the same across chunks.

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
    """Crossfade left_tail into right_head along time dim using the given mode.

    Both tensors must have shape [B, C, T_overlap, H, W]. Returns a new
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
    alpha = alpha.view(1, 1, T, 1, 1)
    return left_tail * (1.0 - alpha) + right_head * alpha


def _concat_latents(latents, overlap_frames, blend_mode="drop"):
    """Stitch latent chunks along the temporal axis (dim=2).

    - blend_mode="drop":   drop the last `overlap_frames` of each preceding chunk
    - blend_mode="linear": crossfade prev tail with next head, linearly
    - blend_mode="cosine": crossfade prev tail with next head, cosine-eased

    For the blend modes, BOTH the trailing N frames of the preceding chunk and
    the leading N frames of the following chunk are consumed by the blend (so
    the seam consumes 2N source frames and emits N blended frames).

    Returns {"samples": stitched_tensor}. Per-chunk noise_mask keys are dropped.
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
        return {"samples": combined, "type": latents[0].get("type", "audio")}

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
    return {"samples": combined}


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

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongVideoStitcher",
            display_name="Long Video Stitcher",
            category="WhatDreamsCost",
            description=(
                "Concatenates up to 6 video latents along the temporal axis. "
                "blend_mode=drop trims the trailing overlap_frames of each preceding "
                "chunk; blend_mode=linear/cosine crossfades the overlap region across "
                "each seam for a smoother boundary. Channels / spatial dims of all "
                "chunks must match. Also accepts 4D audio latents ([B,C,T,F]) — those "
                "are plainly concatenated (overlap_frames / blend_mode are ignored) so "
                "you can stitch the matching audio of chained chunks with a second instance."
            ),
            inputs=[
                io.Latent.Input("latent_1", tooltip="First chunk (required)."),
                io.Latent.Input("latent_2", optional=True, tooltip="Second chunk."),
                io.Latent.Input("latent_3", optional=True, tooltip="Third chunk."),
                io.Latent.Input("latent_4", optional=True, tooltip="Fourth chunk."),
                io.Latent.Input("latent_5", optional=True, tooltip="Fifth chunk."),
                io.Latent.Input("latent_6", optional=True, tooltip="Sixth chunk."),
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
                latent_5=None, latent_6=None, overlap_frames=0,
                blend_mode="drop") -> io.NodeOutput:
        chunks = [l for l in (latent_1, latent_2, latent_3, latent_4, latent_5, latent_6)
                  if l is not None]
        out = _concat_latents(chunks, int(overlap_frames), blend_mode)
        return io.NodeOutput(out)


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


class LongChunkSampler(io.ComfyNode):
    """Run KSampler N times to produce a long video latent in one node.

    Each chunk after the first has its leading `seed_overlap_latent_frames`
    locked to the previous chunk's tail (via a noise_mask = 0 on those
    frames), so the sampler must denoise the remainder consistent with the
    seeded prefix. The stitched output drops the seeded prefix of each
    chunk after the first so the seam doesn't double up.

    Best fit: "extend the same shot for N × chunk_length" workflows where
    the same conditioning (positive / negative / model) applies across all
    chunks. For per-chunk prompt evolution chain Director + KSampler
    manually and use LongVideoStitcher to combine.

    Notes:
    - The conditioning is identical for every chunk, including any
      prompt-relay attention mask that was patched onto the model. The
      latent shape must match the shape the mask was built for, which is
      automatic when the input `latent` came straight from the same
      Director that patched the model.
    - For Wan I2V/FLF this still re-anchors each chunk on the original
      start_image (because that's baked into the positive conditioning via
      concat_latent_image). Use only for hold-style extensions on Wan.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongChunkSampler",
            display_name="Long Chunk Sampler",
            category="WhatDreamsCost",
            description=(
                "Runs KSampler num_chunks times, seeding each chunk after the first "
                "from the previous chunk's tail (locked via noise_mask), and outputs "
                "a single stitched latent. Same conditioning for every chunk; for "
                "per-chunk prompt evolution chain Director + KSampler manually."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent", tooltip="Chunk-length latent (output of a Director)."),
                io.Int.Input(
                    "num_chunks", default=3, min=1, max=20, step=1,
                    tooltip="How many chunks to generate and stitch. Total output length = chunk_length × num_chunks − (num_chunks − 1) × seed_overlap_latent_frames.",
                ),
                io.Int.Input(
                    "seed_overlap_latent_frames", default=1, min=1, max=16, step=1,
                    tooltip="Latent frames at the front of each chunk (after the first) that are locked to the previous chunk's tail. Drives continuity at every seam.",
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=200, step=1,
                ),
                io.Float.Input(
                    "cfg", default=3.0, min=0.0, max=20.0, step=0.1,
                ),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    tooltip="Base seed. Each chunk uses seed + chunk_index so they don't all collapse to the same noise pattern.",
                ),
                io.Float.Input(
                    "denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent,
                num_chunks=3, seed_overlap_latent_frames=1,
                steps=20, cfg=3.0, sampler_name="euler", scheduler="normal",
                seed=0, denoise=1.0) -> io.NodeOutput:
        base_samples = latent["samples"]
        if base_samples.dim() != 5:
            raise ValueError(
                f"LongChunkSampler: expected a 5D video latent [B,C,T,H,W], got shape "
                f"{tuple(base_samples.shape)}."
            )

        base_T = base_samples.shape[2]
        overlap = max(1, int(seed_overlap_latent_frames))
        if overlap >= base_T:
            raise ValueError(
                f"LongChunkSampler: seed_overlap_latent_frames ({overlap}) must be smaller "
                f"than the latent's temporal length ({base_T})."
            )

        num_chunks = max(1, int(num_chunks))
        all_chunks = []

        for chunk_idx in range(num_chunks):
            chunk_seed = (int(seed) + chunk_idx) & 0xffffffffffffffff

            if chunk_idx == 0:
                chunk_latent = latent
            else:
                prev_samples = all_chunks[-1]["samples"]
                new_samples = base_samples.clone()
                copy = min(overlap, prev_samples.shape[2])
                new_samples[:, :, :copy] = prev_samples[:, :, -copy:]
                # noise_mask=0 means "preserve" (don't denoise this region) for the
                # ComfyUI sampler. Shape [B,1,T,1,1] broadcasts across spatial dims.
                mask = torch.ones(
                    (new_samples.shape[0], 1, new_samples.shape[2], 1, 1),
                    dtype=new_samples.dtype, device=new_samples.device,
                )
                mask[:, :, :copy] = 0.0
                chunk_latent = {"samples": new_samples, "noise_mask": mask}

            log.info(
                "[LongChunkSampler] Chunk %d/%d: seed=%d, latent shape=%s",
                chunk_idx + 1, num_chunks, chunk_seed,
                tuple(chunk_latent["samples"].shape),
            )

            sampled = nodes.common_ksampler(
                model, chunk_seed, int(steps), float(cfg), sampler_name, scheduler,
                positive, negative, chunk_latent, denoise=float(denoise),
            )[0]
            all_chunks.append(sampled)

        # Stitch: drop the seeded prefix from every chunk after the first so the
        # overlap doesn't double up. Chunk 0 contributes its full latent.
        stitched_parts = [all_chunks[0]["samples"]]
        for chunk in all_chunks[1:]:
            s = chunk["samples"]
            if overlap < s.shape[2]:
                stitched_parts.append(s[:, :, overlap:])
            else:
                stitched_parts.append(s)

        final = torch.cat(stitched_parts, dim=2)
        log.info(
            "[LongChunkSampler] Final stitched latent: %s (chunks=%d, per-chunk T=%d, overlap=%d)",
            tuple(final.shape), num_chunks, base_T, overlap,
        )
        return io.NodeOutput({"samples": final})


# ---------------------------------------------------------------------------
# Phase 3 — Lightning LoRA preset + per-chunk-prompt sampler
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
    LongChunkSampler) so you don't have to remember the magic numbers.

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
                "Wire steps/cfg/sampler_name/scheduler into KSampler or LongChunkSampler."
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


class LongChunkSamplerMulti(io.ComfyNode):
    """Per-chunk-prompt variant of LongChunkSampler.

    Accepts up to 6 positive conditionings (one per chunk) plus a single shared
    negative. The sampler runs once per provided positive, seeding each chunk
    after the first from the previous chunk's tail latent (the leading
    `seed_overlap_latent_frames` are locked via noise_mask=0). num_chunks is
    derived from the number of positives connected.

    Use this when each chunk should be driven by a different prompt (so each
    Director produces its own positive). The model is shared across all
    chunks — the model's prompt-relay mask must therefore be valid for the
    chunk-length latent of every chunk. Easiest way: use the same chunk_length
    timeline_data in every Director.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongChunkSamplerMulti",
            display_name="Long Chunk Sampler (Multi-Prompt)",
            category="WhatDreamsCost",
            description=(
                "Runs KSampler once per connected positive conditioning, seeding each "
                "chunk after the first from the previous chunk's tail. For per-chunk "
                "prompt evolution. num_chunks = number of positive inputs connected."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("negative", tooltip="Negative conditioning, shared across every chunk."),
                io.Latent.Input("latent", tooltip="Chunk-length latent template (output of a Director)."),
                io.Conditioning.Input("positive_1", tooltip="Chunk 1 positive (required)."),
                io.Conditioning.Input("positive_2", optional=True, tooltip="Chunk 2 positive."),
                io.Conditioning.Input("positive_3", optional=True, tooltip="Chunk 3 positive."),
                io.Conditioning.Input("positive_4", optional=True, tooltip="Chunk 4 positive."),
                io.Conditioning.Input("positive_5", optional=True, tooltip="Chunk 5 positive."),
                io.Conditioning.Input("positive_6", optional=True, tooltip="Chunk 6 positive."),
                io.Int.Input(
                    "seed_overlap_latent_frames", default=1, min=1, max=16, step=1,
                    tooltip="Latent frames at the front of each chunk after the first that are locked to the previous chunk's tail.",
                ),
                io.Int.Input("steps", default=20, min=1, max=200, step=1),
                io.Float.Input("cfg", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    tooltip="Base seed. Chunk K uses seed + K.",
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, negative, latent, positive_1,
                positive_2=None, positive_3=None, positive_4=None,
                positive_5=None, positive_6=None,
                seed_overlap_latent_frames=1,
                steps=20, cfg=3.0, sampler_name="euler", scheduler="normal",
                seed=0, denoise=1.0) -> io.NodeOutput:
        positives = [p for p in (positive_1, positive_2, positive_3, positive_4,
                                 positive_5, positive_6) if p is not None]
        num_chunks = len(positives)

        base_samples = latent["samples"]
        if base_samples.dim() != 5:
            raise ValueError(
                f"LongChunkSamplerMulti: expected a 5D video latent [B,C,T,H,W], got "
                f"shape {tuple(base_samples.shape)}."
            )

        base_T = base_samples.shape[2]
        overlap = max(1, int(seed_overlap_latent_frames))
        if overlap >= base_T:
            raise ValueError(
                f"LongChunkSamplerMulti: seed_overlap_latent_frames ({overlap}) must be "
                f"smaller than the latent's temporal length ({base_T})."
            )

        all_chunks = []
        for chunk_idx, positive in enumerate(positives):
            chunk_seed = (int(seed) + chunk_idx) & 0xffffffffffffffff

            if chunk_idx == 0:
                chunk_latent = latent
            else:
                prev_samples = all_chunks[-1]["samples"]
                new_samples = base_samples.clone()
                copy = min(overlap, prev_samples.shape[2])
                new_samples[:, :, :copy] = prev_samples[:, :, -copy:]
                mask = torch.ones(
                    (new_samples.shape[0], 1, new_samples.shape[2], 1, 1),
                    dtype=new_samples.dtype, device=new_samples.device,
                )
                mask[:, :, :copy] = 0.0
                chunk_latent = {"samples": new_samples, "noise_mask": mask}

            log.info(
                "[LongChunkSamplerMulti] Chunk %d/%d: seed=%d, latent shape=%s",
                chunk_idx + 1, num_chunks, chunk_seed,
                tuple(chunk_latent["samples"].shape),
            )

            sampled = nodes.common_ksampler(
                model, chunk_seed, int(steps), float(cfg), sampler_name, scheduler,
                positive, negative, chunk_latent, denoise=float(denoise),
            )[0]
            all_chunks.append(sampled)

        stitched_parts = [all_chunks[0]["samples"]]
        for chunk in all_chunks[1:]:
            s = chunk["samples"]
            if overlap < s.shape[2]:
                stitched_parts.append(s[:, :, overlap:])
            else:
                stitched_parts.append(s)

        final = torch.cat(stitched_parts, dim=2)
        log.info(
            "[LongChunkSamplerMulti] Final stitched latent: %s (chunks=%d, per-chunk T=%d, overlap=%d)",
            tuple(final.shape), num_chunks, base_T, overlap,
        )
        return io.NodeOutput({"samples": final})


# ---------------------------------------------------------------------------
# Dynamic N-chunk sampler — one prompt per chunk via a multiline list
# ---------------------------------------------------------------------------


def _sample_and_stitch_chunks(model, positives, negative, base_latent, overlap,
                              steps, cfg, sampler_name, scheduler, seed, denoise):
    """Sample one chunk per conditioning in `positives`, seeding each chunk after
    the first from the previous chunk's tail (front `overlap` latent frames locked
    via noise_mask=0), then stitch by dropping each chunk's seeded prefix.

    Shared by the dynamic LongChainSampler; mirrors the inline loop in
    LongChunkSampler / LongChunkSamplerMulti. Returns {"samples": stitched}.
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

    Unlike LongChunkSamplerMulti (capped at 6 conditioning sockets), the per-chunk
    prompts are a multiline list — one prompt per line (`|` also separates) — so the
    chunk count is unbounded. `num_chunks` drives how many chunks are generated:

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
