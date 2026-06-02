"""Generate a WanDirector (v2) example workflow.

    python build_wan_director_example.py {fp8|gguf} OUT.json

Builds a clean, loadable single-segment Wan Director graph:

    UNET(fp8) | UnetLoaderGGUF  -> lightx2v LoRA -> ModelSamplingSD3 --\
                                                                        Wan Director -> KSamplerAdvanced(high) -> (low) -> VAEDecode -> VHS_VideoCombine
    (same for the low-noise model) --------------------------------- /     |  |
                                                       CLIPLoader -------/  |
                                                       VAELoader ----------+--+ (vae also -> VAEDecode)

Loader / sampler / decode node STRUCTURES are cloned from real reference
workflows (guaranteed-valid); the WanDirector node is hand-built to match the v2
schema with the correct REQUIRED-then-OPTIONAL widget order.

Kept intentionally clean: only the public lightx2v speed LoRA (needed for the
low-step MoE sampling). Add your own content LoRAs + a start image / prompt in
the timeline. Timeline starts empty — open the node and add segments.
"""
import copy
import json
import sys

FP8_REF = "video_wan2_2_14B_i2v_FLframe_transition NeurWish.json"
GGUF_REF = "Wan2.2_I2V_SVI_Workflow_Kenpechi_v3.5.json"

MODE = sys.argv[1] if len(sys.argv) > 1 else "fp8"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"Wan Director ({MODE}) v1.json"
assert MODE in ("fp8", "gguf"), "mode must be fp8 or gguf"

fp8 = json.load(open(FP8_REF, encoding="utf-8"))
gguf = json.load(open(GGUF_REF, encoding="utf-8"))


def first_of(doc, t):
    for n in doc["nodes"]:
        if n.get("type") == t:
            return n
    raise SystemExit(f"reference has no {t}")


_nid = [0]
_lid = [0]
out_nodes = []
out_links = []


def nid():
    _nid[0] += 1
    return _nid[0]


def lid():
    _lid[0] += 1
    return _lid[0]


def clone(doc, type_, pos, widgets=None):
    n = copy.deepcopy(first_of(doc, type_))
    n["id"] = nid()
    n["pos"] = list(pos)
    n["flags"] = {}
    n["order"] = 0
    n["mode"] = 0
    for inp in n.get("inputs", []) or []:
        inp["link"] = None
    for o in n.get("outputs", []) or []:
        o["links"] = []
    if widgets is not None:
        n["widgets_values"] = widgets
    out_nodes.append(n)
    return n


def out_slot(node, name):
    for i, o in enumerate(node.get("outputs", []) or []):
        if o.get("name") == name:
            return i
    return 0


def in_slot(node, name):
    for i, inp in enumerate(node.get("inputs", []) or []):
        if inp.get("name") == name:
            return i
    return None


def connect(src_node, src_name, dst_node, dst_name, typ):
    si = out_slot(src_node, src_name)
    di = in_slot(dst_node, dst_name)
    if di is None:
        raise SystemExit(f"no input {dst_name!r} on {dst_node['type']}")
    L = lid()
    out_links.append([L, src_node["id"], si, dst_node["id"], di, typ])
    dst_node["inputs"][di]["link"] = L
    o = src_node["outputs"][si]
    o["links"] = (o.get("links") or []) + [L]
    return L


# ---- per-mode model config ------------------------------------------------
LIGHTX2V = "wan2.2\\Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
if MODE == "fp8":
    LOADER_TYPE = "UNETLoader"
    UNET_HIGH = ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "fp8_e4m3fn_fast"]
    UNET_LOW = ["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "fp8_e4m3fn_fast"]
    SD3_SHIFT = 8.0
    STEPS, CFG, SAMPLER, SCHED, MOE_BOUNDARY = 8, 1.0, "euler", "simple", 4
else:  # gguf
    LOADER_TYPE = "UnetLoaderGGUF"
    UNET_HIGH = ["Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf"]
    UNET_LOW = ["Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"]
    SD3_SHIFT = 5.0
    STEPS, CFG, SAMPLER, SCHED, MOE_BOUNDARY = 8, 1.0, "euler", "simple", 3
CHUNK_FRAMES = 81

PUSA_HIGH = "wan2.2\\Wan22_PusaV1_lora_HIGH_resized_dynamic_avg_rank_98_bf16.safetensors"
PUSA_LOW = "wan2.2\\Wan22_PusaV1_lora_LOW_resized_dynamic_avg_rank_98_bf16.safetensors"
# lightx2v = low-step speed LoRA. Standard strength ~1.0 — at the FLF reference's
# 5.0/2.0 the step-distill LoRA dominates and the model ignores fine input-image
# conditioning (the start image drifts). 1.0 keeps low-step sampling while staying
# faithful to the image. PusaV1 = I2V fidelity (helps keep the input image).
LORA_HIGH = [(LIGHTX2V, 1.0), (PUSA_HIGH, 1.5)]
LORA_LOW = [(LIGHTX2V, 1.0), (PUSA_LOW, 1.3)]
CLIP_VISION_NAME = "clip_vision_h.safetensors"
CLIP_W = ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"]
VAE_W = ["wan_2.1_vae.safetensors"] if MODE == "gguf" else ["wan 2.1\\wan_2.1_vae_Comfy-Org.safetensors"]


def build_model_chain(unet_widgets, loras, y):
    x = 0
    loader_doc = gguf if LOADER_TYPE == "UnetLoaderGGUF" else fp8
    unet = clone(loader_doc, LOADER_TYPE, (x, y), list(unet_widgets))
    prev = unet
    x += 320
    for name, strength in loras:
        lora = clone(fp8, "LoraLoaderModelOnly", (x, y), [name, strength])
        connect(prev, "MODEL", lora, "model", "MODEL")
        prev = lora
        x += 320
    sd3 = clone(fp8, "ModelSamplingSD3", (x, y), [SD3_SHIFT])
    connect(prev, "MODEL", sd3, "model", "MODEL")
    return sd3


sd3_high = build_model_chain(UNET_HIGH, LORA_HIGH, 0)
sd3_low = build_model_chain(UNET_LOW, LORA_LOW, 560)
clip = clone(fp8, "CLIPLoader", (0, 1120), CLIP_W)
vae = clone(fp8, "VAELoader", (0, 1280), VAE_W)
clip_vision = clone(fp8, "CLIPVisionLoader", (0, 1400), [CLIP_VISION_NAME])

# ---- Wan Director (v2 schema, hand-built) ---------------------------------
director = {
    "id": nid(), "type": "WanDirector", "pos": [1700, 200], "size": [1000, 760],
    "flags": {}, "order": 0, "mode": 0,
    "inputs": [
        {"name": "model_high", "type": "MODEL", "link": None},
        {"name": "model_low", "type": "MODEL", "shape": 7, "link": None},
        {"name": "clip", "type": "CLIP", "link": None},
        {"name": "vae", "type": "VAE", "link": None},
        {"name": "clip_vision", "type": "CLIP_VISION", "shape": 7, "link": None},
        {"name": "clip_vision_start", "type": "CLIP_VISION_OUTPUT", "shape": 7, "link": None},
        {"name": "clip_vision_end", "type": "CLIP_VISION_OUTPUT", "shape": 7, "link": None},
    ],
    "outputs": [
        {"name": "latent", "type": "LATENT", "links": []},
        {"name": "frame_rate", "type": "FLOAT", "links": []},
    ],
    "properties": {"cnr_id": "whatdreamscost-comfyui", "Node name for S&R": "WanDirector"},
    # v3 widget order: REQUIRED (schema order) then OPTIONAL (schema order).
    #   required: global_prompt, global_negative_prompt, width, height, length, batch_size,
    #             duration_frames, duration_seconds, timeline_data, local_prompts,
    #             segment_lengths, epsilon, steps, cfg, sampler_name, scheduler, seed,
    #             chunk_frames, moe_boundary
    #   optional: i2v_backend, frame_rate, display_mode, divisible_by, max_side,
    #             guide_strength, keyframe_hold, use_prompt_relay, colormatch_strength
    # width/height/length = 0 -> auto (length follows the timeline). max_side caps the
    # longest side (~480p default) for speed. keyframe_hold pins each reference image
    # across N frames for stricter adherence (5 ≈ 2 latent frames). Director chains
    # segment-aligned FLF clips in latent space (MoE high->low) and outputs one LATENT.
    "widgets_values": [
        "", "", 0, 0, 0, 1, 120, 5.0, "", "", "", 0.001,
        STEPS, CFG, SAMPLER, SCHED, 0, CHUNK_FRAMES, MOE_BOUNDARY,
        "native", 16, "seconds", 16, 832, "", 5, False, 0.0, False,
    ],
}
out_nodes.append(director)

connect(sd3_high, "MODEL", director, "model_high", "MODEL")
connect(sd3_low, "MODEL", director, "model_low", "MODEL")
connect(clip, "CLIP", director, "clip", "CLIP")
connect(vae, "VAE", director, "vae", "VAE")
connect(clip_vision, "CLIP_VISION", director, "clip_vision", "CLIP_VISION")

# Director chains FLF clips in latent space -> one stitched LATENT -> VAEDecode -> VideoCombine.
vdec = clone(fp8, "VAEDecode", (2800, 100))
connect(director, "latent", vdec, "samples", "LATENT")
connect(vae, "VAE", vdec, "vae", "VAE")
vcombine = clone(fp8, "VHS_VideoCombine", (3080, 100))
connect(vdec, "IMAGE", vcombine, "images", "IMAGE")

wf = {
    "id": f"wan-director-{MODE}", "revision": 0,
    "last_node_id": _nid[0], "last_link_id": _lid[0],
    "nodes": out_nodes, "links": out_links,
    "groups": [], "config": {}, "extra": {}, "version": 0.4,
}
json.dump(wf, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"done [{MODE}] -> {OUT} | nodes={len(out_nodes)} links={len(out_links)}")
print("types:", sorted(set(n["type"] for n in out_nodes)))
