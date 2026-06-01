"""Wan model data-processing helpers for the Wan Director ecosystem (Phase 1).

The whole point of "make a Wan version" is the model-processing logic, not just a
node. Rather than re-deriving Wan's latent geometry and I2V/FLF conditioning by
hand (brittle across ComfyUI / VAE versions), we DELEGATE to ComfyUI's native Wan
nodes — exactly how the LTX Director reuses comfy's ``LTXVAddGuide``. That keeps
our latent + conditioning byte-for-byte compatible with the rest of the Wan
ecosystem (samplers, GGUF loaders, etc.).

Phase 0 decision (Redmine #92): the I2V backend is switchable —
  * ``native`` (default): comfy_extras ``WanImageToVideo`` / ``WanFirstLastFrameToVideo``
  * ``fmlf``  (optional): ComfyUI-Wan22FMLF ``WanAdvancedI2V`` (SVI / prev_latent);
    used mainly for the deferred long-video chaining (Phase 6).

Pure helpers (``pixel_to_latent_len``, ``snap``, ``resolve_wan_dims``) avoid
importing torch/comfy at module load so they can be unit-tested off-GPU; the
encode helpers lazy-import comfy inside the function body.
"""
import logging

log = logging.getLogger(__name__)

# Wan VAE temporal compression. Matches WanImageToVideo's ((length - 1) // 4) + 1
# and patches.detect_model_type()'s temporal stride of 4 for arch == "wan".
WAN_TEMPORAL_STRIDE = 4
# Fallback spatial compression / latent channels for the 14B / Wan2.1-VAE path.
# Both are derived from the VAE at runtime when possible (5B uses 16 / 48).
_DEFAULT_SPATIAL = 8
_DEFAULT_CHANNELS = 16


def pixel_to_latent_len(pixel_len, stride=WAN_TEMPORAL_STRIDE):
    """Pixel-space frame count -> Wan latent frame count: ``((n - 1) // stride) + 1``.

    Identical to the formula baked into WanImageToVideo, so latent_t computed here
    matches the latent the native node produces.
    """
    pixel_len = int(pixel_len)
    if pixel_len < 1:
        return 1
    return ((pixel_len - 1) // stride) + 1


def latent_to_pixel_len(latent_len, stride=WAN_TEMPORAL_STRIDE):
    """Inverse of :func:`pixel_to_latent_len` (smallest pixel length for a latent_t)."""
    latent_len = int(latent_len)
    if latent_len < 1:
        return 1
    return (latent_len - 1) * stride + 1


def snap(val, div):
    """Snap ``val`` down to a multiple of ``div`` (never below ``div``)."""
    div = max(1, int(div))
    return max(div, (int(val) // div) * div)


def segment_latent_lengths(pixel_lengths, latent_frames, stride=WAN_TEMPORAL_STRIDE):
    """Convert per-segment pixel lengths to latent-frame lengths, clamped so the
    running total never exceeds ``latent_frames``. Returns None if input is empty.

    The Prompt Relay segmenter (prompt_relay.distribute_segment_lengths) consumes
    latent-frame counts; the timeline editor produces pixel-frame counts.
    """
    if not pixel_lengths:
        return None
    out = []
    used = 0
    for p in pixel_lengths:
        n = pixel_to_latent_len(p, stride)
        n = max(0, min(n, latent_frames - used))
        out.append(n)
        used += n
    return out


def resolve_wan_dims(start_image, width, height, divisible_by=16, max_side=0):
    """Resolve target (width, height). 0 = auto-detect from the start image's native
    size, snapped to ``divisible_by`` (LTX-style convenience).

    ``max_side`` (> 0) caps the longest dimension, scaling both down while keeping
    aspect ratio. This is the main speed lever: a 1072x1600 image runs ~4x slower
    than Wan's ~480p native, so capping to e.g. 832 (≈480p) or 1280 (≈720p) keeps
    generation fast and on-distribution.

    ``start_image`` is a ComfyUI IMAGE tensor [B, H, W, C] (only ``.shape`` is read,
    so no torch import is needed). Raises if no image and a dimension is still 0.
    """
    if start_image is not None:
        nat_h, nat_w = int(start_image.shape[1]), int(start_image.shape[2])
        w = width if width and width > 0 else nat_w
        h = height if height and height > 0 else nat_h
    else:
        w, h = width, height
    if not w or not h or w <= 0 or h <= 0:
        raise ValueError(
            "Wan Director: no start image to auto-detect dimensions from. Set width "
            "and height explicitly (T2V), or provide a start-image segment."
        )
    if max_side and max_side > 0 and max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))
    return snap(w, divisible_by), snap(h, divisible_by)


def _vae_spatial_scale(vae):
    """Spatial downscale factor of the Wan VAE (8 for 2.1, 16 for 2.2-5B)."""
    fn = getattr(vae, "spacial_compression_encode", None)  # ComfyUI spelling
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    return _DEFAULT_SPATIAL


def _vae_channels(vae):
    return int(getattr(vae, "latent_channels", _DEFAULT_CHANNELS) or _DEFAULT_CHANNELS)


def build_wan_latent(vae, width, height, length, batch_size=1, device=None):
    """Empty Wan video latent, shape-matched to the native node:
    ``[batch, channels, ((length-1)//4)+1, height // sp, width // sp]``.

    channels / spatial scale are read from the VAE so this works for both the 14B
    (16ch, /8) and 5B (48ch, /16) families. Returns ``(latent_dict, latent_t)``.
    """
    import torch
    import comfy.model_management

    if device is None:
        device = comfy.model_management.intermediate_device()
    latent_t = pixel_to_latent_len(length)
    sp = _vae_spatial_scale(vae)
    ch = _vae_channels(vae)
    samples = torch.zeros(
        [int(batch_size), ch, latent_t, int(height) // sp, int(width) // sp],
        device=device,
    )
    return {"samples": samples}, latent_t


def encode_wan_i2v(positive, negative, vae, width, height, length, batch_size=1,
                   start_image=None, end_image=None,
                   clip_vision_start=None, clip_vision_end=None,
                   backend="native"):
    """Build Wan latent + I2V/FLF/T2V conditioning by delegating to the real Wan nodes.

    Returns ``(positive, negative, latent_dict, latent_t)``.

    Routing:
      * ``end_image`` given        -> WanFirstLastFrameToVideo  (FLF)
      * ``start_image`` only       -> WanImageToVideo            (I2V)
      * neither                    -> WanImageToVideo w/ start_image=None (T2V empty latent)
      * ``backend == "fmlf"``      -> ComfyUI-Wan22FMLF WanAdvancedI2V if installed,
                                      else log + fall back to native.
    """
    if backend == "fmlf":
        result = _encode_fmlf(positive, negative, vae, width, height, length, batch_size,
                              start_image, end_image, clip_vision_start, clip_vision_end)
        if result is not None:
            pos, neg, latent = result
            return pos, neg, latent, int(latent["samples"].shape[2])
        log.warning("[WanDirector] i2v_backend='fmlf' unavailable; falling back to native.")

    from comfy_extras.nodes_wan import WanImageToVideo, WanFirstLastFrameToVideo

    if end_image is not None:
        out = WanFirstLastFrameToVideo.execute(
            positive, negative, vae, width, height, length, batch_size,
            start_image=start_image, end_image=end_image,
            clip_vision_start_image=clip_vision_start,
            clip_vision_end_image=clip_vision_end,
        )
    else:
        out = WanImageToVideo.execute(
            positive, negative, vae, width, height, length, batch_size,
            start_image=start_image, clip_vision_output=clip_vision_start,
        )
    pos, neg, latent = out.result
    return pos, neg, latent, int(latent["samples"].shape[2])


def _encode_fmlf(positive, negative, vae, width, height, length, batch_size,
                 start_image, end_image, clip_vision_start, clip_vision_end):
    """Optional Wan22FMLF backend (WanAdvancedI2V). Returns (pos, neg, latent) or
    None if the custom node pack isn't installed / its API doesn't match.

    NOTE: the exact WanAdvancedI2V signature is finalised in Phase 6 (long-video
    SVI). For now this guards the import so selecting 'fmlf' degrades gracefully to
    native instead of crashing.
    """
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-Wan22FMLF")  # placeholder path
    except Exception:
        return None
    # Phase 6 will wire WanAdvancedI2V here (prev_latent / svi_motion_strength).
    log.info("[WanDirector] fmlf backend located but not yet wired (Phase 6).")
    return None
