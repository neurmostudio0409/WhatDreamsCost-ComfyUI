"""Generate a clean Wan Director workflow JSON, reusing the model setup from a
reference Wan 2.2 14B MoE FLF workflow.

    python build_wan_director_workflow.py REF.json OUT.json

What it builds (single coherent Wan Director chunk):

    UNETLoader(high) -> 3 LoRAs -> ModelSamplingSD3 ----\
                                                         Wan Director -> KSamplerAdvanced(high) -> KSamplerAdvanced(low) -> VAEDecode -> VHS_VideoCombine
    UNETLoader(low)  -> 3 LoRAs -> ModelSamplingSD3 ----/        |  |  |
                                                  CLIPLoader ----/  |  |
                                                  VAELoader --------+--+ (vae also -> VAEDecode)

Loader filenames, LoRA names/strengths, ModelSamplingSD3 shift and the two
KSamplerAdvanced settings are all CLONED from the reference, so the models match
your setup. The Wan Director timeline starts empty - open it in ComfyUI and add
your start image + prompt segments (and an end image for FLF).

Note: the KJ perf/quality patch nodes (SageAttention / WanVideoNAG / CFGZeroStar)
from the reference are intentionally left out to keep this dependency-light; re-add
them in the model chain if you want them.
"""
import copy
import json
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "video_wan2_2_14B_i2v_FLframe_transition NeurWish.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "Wan Director (single chunk) v1.json"

ref = json.load(open(REF, encoding="utf-8"))
ref_nodes = {n["id"]: n for n in ref["nodes"]}


def first_of(t):
    for n in ref["nodes"]:
        if n.get("type") == t:
            return n
    raise SystemExit("reference has no %s node" % t)


# ---- id / link counters for the NEW graph -------------------------------
_nid = [0]
_lid = [0]


def nid():
    _nid[0] += 1
    return _nid[0]


def lid():
    _lid[0] += 1
    return _lid[0]


out_nodes = []
out_links = []


def clone(template_type, pos, widgets=None):
    """Deep-copy a reference node by type, give it a fresh id/pos, wipe all link
    state so we can rewire from scratch. Optionally override widgets_values."""
    src = first_of(template_type)
    n = copy.deepcopy(src)
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
        raise SystemExit("no input %r on %s" % (dst_name, dst_node["type"]))
    L = lid()
    out_links.append([L, src_node["id"], si, dst_node["id"], di, typ])
    dst_node["inputs"][di]["link"] = L
    o = src_node["outputs"][si]
    o["links"] = (o.get("links") or []) + [L]
    return L


# ---- model chains (cloned UNETLoader + LoRAs + ModelSamplingSD3) ----------
LORA_HIGH = [
    ("wan2.2\\Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors", 5.0),
    ("wan2.2\\WAN-2.2-I2V-POV-Body-Cumshot-Pullout-HIGH-v1.safetensors", 1.3),
    ("wan2.2\\Wan22_PusaV1_lora_HIGH_resized_dynamic_avg_rank_98_bf16.safetensors", 1.5),
]
LORA_LOW = [
    ("wan2.2\\Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors", 2.0),
    ("wan2.2\\WAN-2.2-I2V-POV-Body-Cumshot-Pullout-LOW-v1.safetensors", 1.0),
    ("wan2.2\\Wan22_PusaV1_lora_LOW_resized_dynamic_avg_rank_98_bf16.safetensors", 1.3),
]
UNET_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WEIGHT_DTYPE = "fp8_e4m3fn_fast"
SD3_SHIFT = 8.0


def build_model_chain(unet_name, loras, y):
    x = 0
    unet = clone("UNETLoader", (x, y), [unet_name, WEIGHT_DTYPE])
    prev, prev_out = unet, "MODEL"
    x += 320
    for name, strength in loras:
        lora = clone("LoraLoaderModelOnly", (x, y), [name, strength])
        connect(prev, prev_out, lora, "model", "MODEL")
        prev, prev_out = lora, "MODEL"
        x += 320
    sd3 = clone("ModelSamplingSD3", (x, y), [SD3_SHIFT])
    connect(prev, prev_out, sd3, "model", "MODEL")
    return sd3


sd3_high = build_model_chain(UNET_HIGH, LORA_HIGH, 0)
sd3_low = build_model_chain(UNET_LOW, LORA_LOW, 560)

clip = clone("CLIPLoader", (0, 1120))          # umt5_xxl ... 'wan' 'default'
vae = clone("VAELoader", (0, 1280))            # wan_2.1_vae

# ---- Wan Director (hand-built from schema) --------------------------------
DIR_X = 1700
director = {
    "id": nid(),
    "type": "WanDirector",
    "pos": [DIR_X, 200],
    "size": [400, 520],
    "flags": {},
    "order": 0,
    "mode": 0,
    "inputs": [
        {"name": "model_high", "type": "MODEL", "link": None},
        {"name": "model_low", "type": "MODEL", "shape": 7, "link": None},
        {"name": "clip", "type": "CLIP", "link": None},
        {"name": "vae", "type": "VAE", "shape": 7, "link": None},
        {"name": "prev_latent", "type": "LATENT", "shape": 7, "link": None},
        {"name": "negative", "type": "CONDITIONING", "shape": 7, "link": None},
        {"name": "clip_vision_start", "type": "CLIP_VISION_OUTPUT", "shape": 7, "link": None},
        {"name": "clip_vision_end", "type": "CLIP_VISION_OUTPUT", "shape": 7, "link": None},
    ],
    "outputs": [
        {"name": "model_high", "type": "MODEL", "links": []},
        {"name": "model_low", "type": "MODEL", "links": []},
        {"name": "positive", "type": "CONDITIONING", "links": []},
        {"name": "negative", "type": "CONDITIONING", "links": []},
        {"name": "latent", "type": "LATENT", "links": []},
        {"name": "frame_rate", "type": "FLOAT", "links": []},
    ],
    "properties": {
        "cnr_id": "whatdreamscost-comfyui",
        "Node name for S&R": "WanDirector",
    },
    # widgets_values follow the schema's widget-input declaration order:
    # model_variant, global_prompt, width, height, length, batch_size,
    # duration_frames, duration_seconds, timeline_data, use_custom_audio,
    # local_prompts, segment_lengths, epsilon, frame_rate, display_mode,
    # guide_strength, custom_width, custom_height, resize_method,
    # divisible_by, img_compression
    "widgets_values": [
        "auto", "", 832, 480, 81, 1, 120, 5.0, "", False,
        "", "", 0.001, 24, "seconds", "", 0, 0,
        "maintain aspect ratio", 16, 0,
    ],
}
out_nodes.append(director)

connect(sd3_high, "MODEL", director, "model_high", "MODEL")
connect(sd3_low, "MODEL", director, "model_low", "MODEL")
connect(clip, "CLIP", director, "clip", "CLIP")
connect(vae, "VAE", director, "vae", "VAE")

# ---- MoE sampler pair (cloned KSamplerAdvanced settings) ------------------
# high: ['enable', seed, 'randomize', 6, 1, 'lcm', 'simple', 0, 3, 'enable']
# low:  ['disable', 0, 'fixed', 6, 1, 'lcm', 'simple', 3, 10000, 'disable']
ks_high = clone("KSamplerAdvanced", (2200, 100),
                ["enable", 0, "randomize", 6, 1, "lcm", "simple", 0, 3, "enable"])
ks_low = clone("KSamplerAdvanced", (2560, 100),
               ["disable", 0, "fixed", 6, 1, "lcm", "simple", 3, 10000, "disable"])

connect(director, "model_high", ks_high, "model", "MODEL")
connect(director, "positive", ks_high, "positive", "CONDITIONING")
connect(director, "negative", ks_high, "negative", "CONDITIONING")
connect(director, "latent", ks_high, "latent_image", "LATENT")

connect(director, "model_low", ks_low, "model", "MODEL")
connect(director, "positive", ks_low, "positive", "CONDITIONING")
connect(director, "negative", ks_low, "negative", "CONDITIONING")
connect(ks_high, "LATENT", ks_low, "latent_image", "LATENT")

# ---- decode + video out ---------------------------------------------------
vdec = clone("VAEDecode", (2920, 100))
connect(ks_low, "LATENT", vdec, "samples", "LATENT")
connect(vae, "VAE", vdec, "vae", "VAE")

vcombine = clone("VHS_VideoCombine", (3200, 100))
connect(vdec, "IMAGE", vcombine, "images", "IMAGE")

# ---- assemble workflow ----------------------------------------------------
wf = {
    "id": "wan-director-single-chunk",
    "revision": 0,
    "last_node_id": _nid[0],
    "last_link_id": _lid[0],
    "nodes": out_nodes,
    "links": out_links,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}
json.dump(wf, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("done ->", OUT)
print("nodes:", len(out_nodes), "links:", len(out_links))
print("types:", sorted(set(n["type"] for n in out_nodes)))
