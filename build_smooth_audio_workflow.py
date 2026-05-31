"""Rewire a chained-LTX workflow to use Smooth Audio Join (waveform crossfade audio).

Run in the folder that has your exported workflow:

    python build_smooth_audio_workflow.py            # reads workflow.json -> workflow_fixed.json
    python build_smooth_audio_workflow.py my.json out.json

What it does (non-destructive — only ADDS nodes + reroutes the final video link):
  - Reads the LTX Smooth Transition node's connections to find:
      * its video_latent output
      * its video_vae / audio_vae sources
      * the two chunk AUDIO latent sources feeding audio_latent_1 / audio_latent_2
  - Adds:  VAEDecode(video) , 2x LTXVAudioVAEDecode , Smooth Audio Join , Create Video
  - Wires:  video_latent -> VAEDecode -> CreateVideo.images
            chunk audio latents -> LTXVAudioVAEDecode -> Smooth Audio Join -> CreateVideo.audio
  - Reroutes Save Video to the new Create Video (the old Decode subgraph becomes unused).
The LTX Smooth Transition node is left as-is (its own audio output just goes nowhere useful now).
"""
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "workflow.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "workflow_fixed.json"

wf = json.load(open(SRC, encoding="utf-8"))
nodes = wf["nodes"]
links = wf["links"]


def by_type(t):
    return [n for n in nodes if n.get("type") == t]


def node_by_id(i):
    for n in nodes:
        if n.get("id") == i:
            return n
    return None


def link_by_id(i):
    for l in links:
        if l[0] == i:
            return l
    return None


tr = (by_type("LTXSmoothTransition") or [None])[0]
save = (by_type("SaveVideo") or [None])[0]
if tr is None:
    raise SystemExit("No LTXSmoothTransition node found.")
if save is None:
    raise SystemExit("No SaveVideo node found.")

last_node = wf.get("last_node_id") or max(n["id"] for n in nodes if isinstance(n.get("id"), int))
last_link = wf.get("last_link_id") or max((l[0] for l in links), default=0)


def nid():
    global last_node
    last_node += 1
    return last_node


def lid():
    global last_link
    last_link += 1
    return last_link


def out_slot(node, name):
    for i, o in enumerate(node.get("outputs", []) or []):
        if o.get("name") == name:
            return i
    return None


def in_src(node, name):
    for inp in node.get("inputs", []) or []:
        if inp.get("name") == name and inp.get("link") is not None:
            l = link_by_id(inp["link"])
            if l:
                return l[1], l[2]
    return None


vid_out = out_slot(tr, "video_latent")
vvae = in_src(tr, "video_vae")
avae = in_src(tr, "audio_vae")
a1 = in_src(tr, "audio_latent_1")
a2 = in_src(tr, "audio_latent_2")

if vid_out is None or vvae is None or a1 is None:
    raise SystemExit(
        "Need LTX Smooth Transition with video_latent + video_vae + audio_latent_1 connected.\n"
        f"  video_latent out slot={vid_out}, video_vae src={vvae}, audio_latent_1 src={a1}, "
        f"audio_latent_2 src={a2}, audio_vae src={avae}"
    )


def mk(type_, pos, inputs, outputs, widgets, size=(230, 90)):
    return {
        "id": nid(), "type": type_, "pos": list(pos), "size": list(size), "flags": {},
        "order": 0, "mode": 0, "inputs": inputs, "outputs": outputs,
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": type_},
        "widgets_values": widgets,
    }


n_vdec = mk("VAEDecode", (3720, 4230),
            [{"name": "samples", "type": "LATENT", "link": None},
             {"name": "vae", "type": "VAE", "link": None}],
            [{"name": "IMAGE", "type": "IMAGE", "links": []}], [])

n_adec1 = mk("LTXVAudioVAEDecode", (3720, 4340),
             [{"name": "samples", "type": "LATENT", "link": None},
              {"name": "audio_vae", "type": "VAE", "link": None}],
             [{"name": "Audio", "type": "AUDIO", "links": []}], [])

n_adec2 = mk("LTXVAudioVAEDecode", (3720, 4450),
             [{"name": "samples", "type": "LATENT", "link": None},
              {"name": "audio_vae", "type": "VAE", "link": None}],
             [{"name": "Audio", "type": "AUDIO", "links": []}], [])

n_join = mk("SmoothAudioJoin", (3990, 4400),
            [{"name": "audio_1", "type": "AUDIO", "link": None},
             {"name": "audio_2", "type": "AUDIO", "shape": 7, "link": None},
             {"name": "crossfade_seconds", "type": "FLOAT", "widget": {"name": "crossfade_seconds"}, "link": None},
             {"name": "blend_mode", "type": "COMBO", "widget": {"name": "blend_mode"}, "link": None}],
            [{"name": "audio", "type": "AUDIO", "links": []}], [0.25, "cosine"])

n_cv = mk("CreateVideo", (4250, 4300),
          [{"name": "images", "type": "IMAGE", "link": None},
           {"name": "audio", "type": "AUDIO", "shape": 7, "link": None},
           {"name": "fps", "type": "FLOAT", "widget": {"name": "fps"}, "link": None}],
          [{"name": "VIDEO", "type": "VIDEO", "links": []}], [24])

nodes.extend([n_vdec, n_adec1, n_adec2, n_join, n_cv])


def connect(src_id, src_slot, dst_node, dst_slot, typ):
    L = lid()
    links.append([L, src_id, src_slot, dst_node["id"], dst_slot, typ])
    dst_node["inputs"][dst_slot]["link"] = L
    s = node_by_id(src_id)
    if s is not None:
        o = s["outputs"][src_slot]
        o["links"] = (o.get("links") or []) + [L]
    return L


# video: transition video_latent -> VAEDecode -> CreateVideo.images
connect(tr["id"], vid_out, n_vdec, 0, "LATENT")
connect(vvae[0], vvae[1], n_vdec, 1, "VAE")
connect(n_vdec["id"], 0, n_cv, 0, "IMAGE")

# audio: chunk latents -> LTXVAudioVAEDecode -> Smooth Audio Join -> CreateVideo.audio
connect(a1[0], a1[1], n_adec1, 0, "LATENT")
if avae:
    connect(avae[0], avae[1], n_adec1, 1, "VAE")
connect(n_adec1["id"], 0, n_join, 0, "AUDIO")
if a2:
    connect(a2[0], a2[1], n_adec2, 0, "LATENT")
    if avae:
        connect(avae[0], avae[1], n_adec2, 1, "VAE")
    connect(n_adec2["id"], 0, n_join, 1, "AUDIO")
connect(n_join["id"], 0, n_cv, 1, "AUDIO")

# reroute Save Video to the new Create Video
for slot, inp in enumerate(save.get("inputs", [])):
    if inp.get("name") == "video":
        old = inp.get("link")
        if old is not None:
            ol = link_by_id(old)
            if ol:
                src = node_by_id(ol[1])
                if src and src["outputs"][ol[2]].get("links"):
                    src["outputs"][ol[2]]["links"] = [x for x in src["outputs"][ol[2]]["links"] if x != old]
                links.remove(ol)
        connect(n_cv["id"], 0, save, slot, "VIDEO")
        break

wf["last_node_id"] = last_node
wf["last_link_id"] = last_link
json.dump(wf, open(DST, "w", encoding="utf-8"), ensure_ascii=False)
print("done ->", DST, "| added 5 nodes, audio via Smooth Audio Join (waveform crossfade)")
print("If audio_latent_2 was not connected it only joined chunk 1's audio; connect more chunks the same way.")
