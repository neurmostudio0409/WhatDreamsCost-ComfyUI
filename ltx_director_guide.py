import logging

from comfy_extras.nodes_lt import LTXVAddGuide
import torch
import comfy.utils
import comfy.samplers
import comfy.model_management
import nodes
from comfy_api.latest import io
from .ltx_director import GuideData

log = logging.getLogger(__name__)


class LTXDirectorGuide(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorGuide",
            display_name="LTX Director Guide",
            category="WhatDreamsCost",
            description=(
                "Applies guide images from a Prompt Relay Timeline node at the frame positions "
                "and strengths defined on the timeline. Connect guide_data from the timeline node."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to add guide keyframe info to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to add guide keyframe info to."),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images."),
                io.Latent.Input("latent", tooltip="Video latent — guides are inserted into this latent."),
                GuideData.Input("guide_data", tooltip="Guide data produced by Prompt Relay Encode (Timeline)."),
                io.Float.Input("scale_by", default=1.0, min=0.01, max=8.0, step=0.01, tooltip="Scale the latent by this factor."),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], default="bicubic", tooltip="Method used to upscale/downscale the latent."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with guide frames applied."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, scale_by=1.0, upscale_method="bicubic") -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula

        # Clone latents to avoid mutating upstream nodes
        latent_image = latent["samples"].clone()

        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        # Apply scale factor if not 1.0
        if scale_by != 1.0:
            B, C, F, H, W = latent_image.shape
            width = round(W * scale_by)
            height = round(H * scale_by)
            
            # Reshape to 4D for common_upscale
            latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            latent_resized_4d = comfy.utils.common_upscale(latent_4d, width, height, upscale_method, "disabled")
            latent_image = latent_resized_4d.reshape(B, F, C, height, width).permute(0, 2, 1, 3, 4)

            # Also resize noise mask if it's not a broadcasted mask
            if noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1:
                mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(B * F, 1, H, W)
                mask_resized_4d = comfy.utils.common_upscale(mask_4d, width, height, upscale_method, "disabled")
                noise_mask = mask_resized_4d.reshape(B, F, 1, height, width).permute(0, 2, 1, 3, 4)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        images = guide_data.get("images", [])
        insert_frames = guide_data.get("insert_frames", [])
        strengths = guide_data.get("strengths", [])

        for idx, img_tensor in enumerate(images):
            f_idx = insert_frames[idx] if idx < len(insert_frames) else 0
            strength = strengths[idx] if idx < len(strengths) else 1.0

            image_1, t = cls.encode(vae, latent_width, latent_height, img_tensor, scale_factors)

            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

            assert latent_idx + t.shape[2] <= latent_length, (
                f"Guide image {idx + 1}: conditioning frames exceed the length of the latent sequence."
            )

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, t, strength, scale_factors,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class LTXSmoothTransition(LTXVAddGuide):
    """Splice MODEL-GENERATED First-Last-Frame transitions between LTX video chunks.

    Unlike a stitcher (which can only crossfade/dissolve two already-finished
    chunks), this GENERATES a brand-new transition clip between each adjacent pair:
    it decodes chunk i's last frame and chunk i+1's first frame, treats them as the
    first/last keyframes of a fresh empty latent, and samples a real FLF morph — so
    the model plans coherent motion across the join. The result is concatenated as
    [chunk 1][generated transition][chunk 2]..., giving native-FLF-smooth joins.

    Reuses the same LTXVAddGuide keyframe machinery as LTX Director Guide, so it is
    LTX-specific. Connect chunks to latent_1..N (slots grow) plus the LTX model +
    VIDEO vae + clip. Handles VIDEO latents only (5D, same spatial dims / channels) —
    stitch the matching audio separately with Smooth Audio Stitcher.
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
            node_id="LTXSmoothTransition",
            display_name="LTX Smooth Transition",
            category="WhatDreamsCost",
            description=(
                "Generates a real First-Last-Frame transition between each pair of LTX video "
                "latent chunks and splices it in: [chunk 1][generated transition][chunk 2]... "
                "Each transition is model-generated (coherent motion) so the joins are as "
                "smooth as a native FLF clip — unlike crossfade/seam stitching which only "
                "dissolves. Connect chunks to latent_1..N (slots grow). Needs the LTX model + "
                "VIDEO vae + clip — this node handles VIDEO latents only; stitch the matching "
                "audio separately with Smooth Audio Stitcher (no VAE needed)."
            ),
            inputs=[
                io.Model.Input("model", tooltip="LTX diffusion model used to generate the transitions."),
                io.Clip.Input("clip", tooltip="LTX CLIP used to encode the (optional) transition prompt."),
                io.Vae.Input("video_vae", tooltip="LTX VIDEO VAE (not the audio VAE). Used to decode each chunk's boundary frame and re-encode it as an FLF keyframe. This node never touches audio — stitch audio separately with Smooth Audio Stitcher."),
                *latent_inputs,
                io.Int.Input(
                    "transition_frames", default=25, min=5, max=257, step=1, optional=True,
                    tooltip="Pixel-frame length of each GENERATED transition clip between chunks. Longer = more room for smooth motion.",
                ),
                io.String.Input(
                    "prompt", multiline=True, default="", optional=True,
                    tooltip="Optional text to steer the generated transitions (e.g. scene / style). Blank = unguided morph.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="How hard each transition is anchored to the two boundary frames (1.0 = exact endpoints).",
                ),
                io.Int.Input("steps", default=20, min=1, max=200, step=1, optional=True),
                io.Float.Input("cfg", default=3.0, min=0.0, max=20.0, step=0.1, optional=True),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", optional=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal", optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, latent_1, latent_2=None, latent_3=None, latent_4=None,
                latent_5=None, latent_6=None, latent_7=None, latent_8=None, latent_9=None,
                latent_10=None, latent_11=None, latent_12=None, transition_frames=25,
                prompt="", strength=1.0, steps=20, cfg=3.0, sampler_name="euler",
                scheduler="normal", seed=0) -> io.NodeOutput:
        ordered = [latent_1, latent_2, latent_3, latent_4, latent_5, latent_6,
                   latent_7, latent_8, latent_9, latent_10, latent_11, latent_12]
        chunks = [l for l in ordered if l is not None]
        if not chunks:
            raise ValueError("LTXSmoothTransition: no latents connected.")

        samples = [c["samples"] for c in chunks]
        ref = samples[0]
        if ref.dim() != 5:
            raise ValueError(
                f"LTXSmoothTransition: expected 5D LTX video latents [B,C,T,H,W], got shape "
                f"{tuple(ref.shape)}."
            )
        for i, s in enumerate(samples[1:], start=1):
            if (s.shape[1], s.shape[3], s.shape[4]) != (ref.shape[1], ref.shape[3], ref.shape[4]):
                raise ValueError(
                    f"LTXSmoothTransition: chunk {i + 1} channels/spatial dims {tuple(s.shape)} "
                    f"differ from chunk 1 {tuple(ref.shape)} — all chunks must match."
                )

        # One chunk: nothing to bridge.
        if len(samples) == 1:
            return io.NodeOutput(chunks[0])

        scale_factors = video_vae.downscale_index_formula
        time_scale = scale_factors[0]
        _, C, _, h, w = ref.shape
        latent_w = w * scale_factors[2]
        latent_h = h * scale_factors[1]
        tf_px = int(transition_frames)
        latent_t = ((tf_px - 1) // time_scale) + 1
        device = comfy.model_management.intermediate_device()
        gp = prompt.strip()

        def boundary_image(lat, take_last):
            frame = lat[:, :, -1:].contiguous() if take_last else lat[:, :, :1].contiguous()
            img = video_vae.decode(frame)
            if img.dim() == 5:  # [B,T,H,W,C] -> drop batch
                img = img[0]
            return img[-1:] if take_last else img[:1]  # [1,H,W,3]

        pieces = [samples[0]]
        for i in range(len(samples) - 1):
            a_img = boundary_image(samples[i], take_last=True)       # chunk i's last frame
            b_img = boundary_image(samples[i + 1], take_last=False)  # chunk i+1's first frame

            # Fresh conditioning per transition (append_keyframe accumulates onto it).
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(gp))
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

            latent_image = torch.zeros([1, C, latent_t, h, w], device=device)
            noise_mask = torch.ones([1, 1, latent_t, 1, 1], dtype=torch.float32, device=device)

            # FLF: image A at frame 0, image B at the last pixel frame.
            for img, f_idx in ((a_img, 0), (b_img, tf_px - 1)):
                image_1, t = cls.encode(video_vae, latent_w, latent_h, img, scale_factors)
                frame_idx, latent_idx = cls.get_latent_index(positive, latent_t, len(image_1), f_idx, scale_factors)
                positive, negative, latent_image, noise_mask = cls.append_keyframe(
                    positive, negative, frame_idx, latent_image, noise_mask, t, float(strength), scale_factors,
                )

            trans = nodes.common_ksampler(
                model, (int(seed) + i) & 0xffffffffffffffff, int(steps), float(cfg),
                sampler_name, scheduler, positive, negative,
                {"samples": latent_image, "noise_mask": noise_mask}, denoise=1.0,
            )[0]
            tlat = trans["samples"]
            # Drop the two anchored endpoint latent frames — they duplicate the chunk
            # boundaries we are bridging.
            if tlat.shape[2] > 2:
                tlat = tlat[:, :, 1:-1]
            pieces.append(tlat)
            pieces.append(samples[i + 1])

        combined = torch.cat(pieces, dim=2)
        log.info(
            "[LTXSmoothTransition] %d chunks + %d generated transitions (%d px each) -> %s",
            len(samples), len(samples) - 1, tf_px, tuple(combined.shape),
        )
        return io.NodeOutput({"samples": combined})