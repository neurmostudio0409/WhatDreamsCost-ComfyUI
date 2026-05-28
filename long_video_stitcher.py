"""Long Video Stitcher — utilities for chaining multiple Director/sampler chunks
into a single long video latent.

Two nodes:

- LongVideoStitcher: concatenates up to 6 latent chunks along the temporal axis,
  with optional `overlap_frames` to drop the trailing frames of each preceding
  chunk (use this when chunk K+1 was generated with chunk K's tail frame as a
  start keyframe, so the duplicate frames at the seam should be dropped).

- LatentTailToImage: VAE-decodes the trailing N frames of a latent into a pixel
  image, intended as the start_image for the next chunk's Director timeline.
  This is the "frame_offset" pattern for LTX, which has no native frame_offset
  the way Wan's S2V / Animate / VACE do.

Audio latents are handled separately — these nodes operate on video latents
only. Combine the chunked audio with the final video in a downstream node
(e.g. VHS_VideoCombine) or stitch the audio independently with the Audio
track in LTX Director.
"""

import logging

import torch

from comfy_api.latest import io


log = logging.getLogger(__name__)


def _concat_latents(latents, overlap_frames):
    """Concatenate latent chunks along the temporal axis (dim=2).

    overlap_frames > 0 drops the LAST `overlap_frames` of each preceding chunk
    (every chunk except the last). Use this when chunk K's tail was reused as
    chunk K+1's start so the overlapping frames don't appear twice in the
    stitched output.

    Returns a single latent dict with key "samples". `noise_mask` keys on
    individual chunks are dropped — sampling has already happened.
    """
    if not latents:
        raise ValueError("LongVideoStitcher: no latents to stitch.")

    samples_list = []
    for i, lat in enumerate(latents):
        s = lat["samples"]
        if overlap_frames > 0 and i < len(latents) - 1 and s.shape[2] > overlap_frames:
            s = s[:, :, :-overlap_frames]
        samples_list.append(s)

    # Sanity-check shapes match on B / C / H / W
    ref = samples_list[0]
    for i, s in enumerate(samples_list[1:], start=1):
        if (s.shape[0], s.shape[1], s.shape[3], s.shape[4]) != \
           (ref.shape[0], ref.shape[1], ref.shape[3], ref.shape[4]):
            raise ValueError(
                f"LongVideoStitcher: chunk {i + 1} shape {tuple(s.shape)} does not "
                f"match chunk 1 shape {tuple(ref.shape)} (batch / channels / spatial "
                "dims must agree)."
            )

    combined = torch.cat(samples_list, dim=2)
    log.info(
        "[LongVideoStitcher] Stitched %d chunks: %s -> %s (overlap_frames=%d)",
        len(latents), [tuple(s.shape) for s in samples_list], tuple(combined.shape),
        overlap_frames,
    )
    return {"samples": combined}


class LongVideoStitcher(io.ComfyNode):
    """Concatenate up to 6 video-latent chunks into a single long latent.

    Typical use: run the Director + Sampler N times (each chunk seeded from the
    previous chunk's last frame via `LatentTailToImage`), connect the resulting
    latents in order, and feed the stitched output into a single VAE Decode.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongVideoStitcher",
            display_name="Long Video Stitcher",
            category="WhatDreamsCost",
            description=(
                "Concatenates up to 6 video latents along the temporal axis. "
                "Drop the trailing overlap of each preceding chunk via overlap_frames "
                "if you seeded chunk K+1 from chunk K's last frame (so the seam frame "
                "doesn't appear twice). Channels / spatial dims of all chunks must match."
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
                        "Latent frames to drop from the end of each preceding chunk. "
                        "Use this when chunk K+1 was seeded with chunk K's tail frame "
                        "to avoid the seam appearing twice. 0 = pure concat."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent_1, latent_2=None, latent_3=None, latent_4=None,
                latent_5=None, latent_6=None, overlap_frames=0) -> io.NodeOutput:
        chunks = [l for l in (latent_1, latent_2, latent_3, latent_4, latent_5, latent_6)
                  if l is not None]
        out = _concat_latents(chunks, int(overlap_frames))
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
