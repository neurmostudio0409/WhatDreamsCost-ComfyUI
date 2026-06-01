"""Generate a chained (multi-chunk) Wan Director workflow JSON, reusing the
model setup from a reference Wan 2.2 14B MoE FLF workflow.

    python build_wan_chain_workflow.py REF.json OUT.json [CHUNKS]

Builds CHUNKS coherent Wan Director segments (default 2) that auto-chain:

  shared loaders (UNET high/low + LoRAs + ModelSamplingSD3, CLIP, VAE)
        |
   fan out to every chunk's Wan Director.model_high / model_low / clip / vae
        |
  chunk 1:  Director1 -> KSampler(high) -> KSampler(low) -> latent1 -----------\
  chunk 2:  Director2(prev_latent <- latent1) -> KSampler(high) -> (low) -> latent2
                ...                                                            |
        all chunk latents -> Long Video Stitcher(blend_mode=drop, overlap=1) ->+ -> VAEDecode -> VHS_VideoCombine

Each chunk's Director decodes the previous chunk's last latent frame (via the
shared vae) and uses it as its start image, so segment N continues seamlessly
from where N-1 ended. Long Video Stitcher(drop, overlap_frames=1) trims the
duplicated seam frame and concatenates into one long video latent.

Open each Director in ComfyUI and add prompt segments. Chunk 1 also needs a
start image; later chunks get their start from prev_latent automatically.
"""
import copy
import json
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "video_wan2_2_14B_i2v_FLframe_transition NeurWish.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "Wan Director Long Video (2x Chain) v1.json"
CHUNKS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

ref = json.load(open(REF, encoding="utf-8"))


def first_of(t):
    for n in ref["nodes"]:
        if n.get("type") == t:
            return n
    raise SystemExit("reference has no %s node" % t)


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
    n = copy.deepcopy(first_of(template_type))
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


# ---- shared model setup ---------------------------------------------------
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
    prev = unet
    x += 320
    for name, strength in loras:
        lora = clone("LoraLoaderModelOnly", (x, y), [name, strength])
        connect(prev, "MODEL", lora, "model", "MODEL")
        prev = lora
        x += 320
    sd3 = clone("ModelSamplingSD3", (x, y), [SD3_SHIFT])
    connect(prev, "MODEL", sd3, "model", "MODEL")
    return sd3


sd3_high = build_model_chain(UNET_HIGH, LORA_HIGH, 0)
sd3_low = build_model_chain(UNET_LOW, LORA_LOW, 560)
clip = clone("CLIPLoader", (0, 1120))
vae = clone("VAELoader", (0, 1280))


# ---- Wan Director factory -------------------------------------------------
def make_director(pos):
    n = {
        "id": nid(),
        "type": "WanDirector",
        "pos": list(pos),
        "size": [400, 520],
        "flags": {}, "order": 0, "mode": 0,
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
        "properties": {"cnr_id": "whatdreamscost-comfyui", "Node name for S&R": "WanDirector"},
        # FRONTEND order: required widgets (schema order) then optional widgets
        # (schema order). width/height = 0 -> auto-detect from the start image.
        #   required: model_variant, global_prompt, width, height, length,
        #             batch_size, duration_frames, duration_seconds, timeline_data,
        #             local_prompts, segment_lengths, epsilon, guide_strength
        #   optional: use_custom_audio, frame_rate, display_mode, custom_width,
        #             custom_height, resize_method, divisible_by, img_compression
        "widgets_values": [
            "auto", "", 0, 0, 81, 1, 120, 5.0, "", "", "", 0.001, "",
            False, 24, "seconds", 0, 0, "maintain aspect ratio", 16, 0,
        ],
    }
    out_nodes.append(n)
    return n


# ---- per-chunk: Director + MoE sampler pair -------------------------------
KS_HIGH = ["enable", 0, "randomize", 6, 1, "lcm", "simple", 0, 3, "enable"]
KS_LOW = ["disable", 0, "fixed", 6, 1, "lcm", "simple", 3, 10000, "disable"]

chunk_latents = []   # final low-noise latent node of each chunk
prev_low = None
for c in range(CHUNKS):
    y = c * 900
    d = make_director((1700, 200 + y))
    connect(sd3_high, "MODEL", d, "model_high", "MODEL")
    connect(sd3_low, "MODEL", d, "model_low", "MODEL")
    connect(clip, "CLIP", d, "clip", "CLIP")
    connect(vae, "VAE", d, "vae", "VAE")
    if prev_low is not None:
        connect(prev_low, "LATENT", d, "prev_latent", "LATENT")   # auto-chain

    ks_h = clone("KSamplerAdvanced", (2200, 100 + y), list(KS_HIGH))
    ks_l = clone("KSamplerAdvanced", (2560, 100 + y), list(KS_LOW))
    connect(d, "model_high", ks_h, "model", "MODEL")
    connect(d, "positive", ks_h, "positive", "CONDITIONING")
    connect(d, "negative", ks_h, "negative", "CONDITIONING")
    connect(d, "latent", ks_h, "latent_image", "LATENT")
    connect(d, "model_low", ks_l, "model", "MODEL")
    connect(d, "positive", ks_l, "positive", "CONDITIONING")
    connect(d, "negative", ks_l, "negative", "CONDITIONING")
    connect(ks_h, "LATENT", ks_l, "latent_image", "LATENT")

    chunk_latents.append(ks_l)
    prev_low = ks_l

# ---- Long Video Stitcher (drop seam) --------------------------------------
stitch_inputs = [{"name": "latent_1", "type": "LATENT", "link": None}]
for i in range(2, CHUNKS + 1):
    stitch_inputs.append({"name": f"latent_{i}", "type": "LATENT", "shape": 7, "link": None})
stitcher = {
    "id": nid(), "type": "LongVideoStitcher", "pos": [2950, 100],
    "size": [260, 120], "flags": {}, "order": 0, "mode": 0,
    "inputs": stitch_inputs,
    "outputs": [{"name": "latent", "type": "LATENT", "links": []}],
    "properties": {"cnr_id": "whatdreamscost-comfyui", "Node name for S&R": "LongVideoStitcher"},
    "widgets_values": [1, "drop"],   # overlap_frames=1, blend_mode=drop
}
out_nodes.append(stitcher)
for i, cl in enumerate(chunk_latents, start=1):
    connect(cl, "LATENT", stitcher, f"latent_{i}", "LATENT")

# ---- decode + video out ---------------------------------------------------
vdec = clone("VAEDecode", (3250, 100))
connect(stitcher, "latent", vdec, "samples", "LATENT")
connect(vae, "VAE", vdec, "vae", "VAE")
vcombine = clone("VHS_VideoCombine", (3530, 100))
connect(vdec, "IMAGE", vcombine, "images", "IMAGE")

wf = {
    "id": "wan-director-chain",
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
print("done ->", OUT, "| chunks:", CHUNKS)
print("nodes:", len(out_nodes), "links:", len(out_links))
