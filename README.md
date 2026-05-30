# Overview

This will be a collection of free resources for ComfyUI.

Hopefully it will make creating cool stuff easier.

All of my nodes are created with the help of AI, so there may or may not be redundant, messy code.

## ▶️ YouTube Tutorial Videos

<table>
  <tr>
    <td>
      <p align="center">LTX Director Trailer</p>
      <a href="https://www.youtube.com/watch?v=fZgtkRcu4_k">
        <img src="https://img.youtube.com/vi/fZgtkRcu4_k/0.jpg" alt="LTX Director Trailer" width="400">
      </a>
    </td>
    <td>
      <p align="center">LTX Director Tutorial</p>
      <a href="https://www.youtube.com/watch?v=vM60pJJqqEI">
        <img src="https://img.youtube.com/vi/vM60pJJqqEI/0.jpg" alt="LTX Director Tutorial" width="400">
      </a>
    </td>
  </tr>
</table>

## ❓ How to install nodes

- Navigate to your `/ComfyUI/custom_nodes/ folder`
- Run `git clone https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI`
- Or download through the ComfyUI Manager.

**❗❗IMPORTANT❗❗**

If you don't see the latest version (v1.3.9) yet in the manager then just downloaded the nightly version (or fetch the updates to update the list to see the latest version). 
Also you will need to update ComfyUI-LTXVideo and ComfyUI-KJNodes to the latest version as well. You cannot use this node without updating ComfyUI-LTXVideo!

# 🔄 Recent Updates
**v1.19.0**
  * **New node: `Smooth Video Stitcher`.** Joins video-latent chunks like `Long Video Stitcher`, but instead of dropping/crossfading frames at each seam (which reads as a cut or a dissolve) it concatenates the chunks and **re-samples a small band of frames at every seam with the model**, so the boundary is a *generated* transition that flows from one chunk into the next. Connect your chunk latents to `latent_1..N` (slots grow as you wire them) and add a plain `model` + `positive`/`negative`; tune `transition_frames` (how wide the regenerated band is) and `denoise` (how strongly it's regenerated). Video latents only — for audio keep using `Long Video Stitcher`.

**v1.18.0**
  * **Removed the image-segment Loop / repeat feature.** The per-segment `Loop` count (the `×N` badge + dashed cycle separators that held one image across N back-to-back cycles) is gone from the timeline UI and from the LTX / Wan backends. To hold an image for longer, just drag its segment edge to the length you want. Old saved workflows still load — any stored `loopCount` is simply ignored (the segment plays once at its base length).

**v1.17.0**
  * **Fixed `prev_latent` auto-chaining (LTX & Wan Director).** Chaining now decodes the previous chunk's **last frame** and uses it as the next chunk's frame-0 start keyframe — exactly what wiring `Latent Tail to Image → start_image` by hand produces. The v1.13.0 "multi-frame motion clip" path could silently fail when the downstream guide was applied (the build error was swallowed), so `prev_latent` appeared to do nothing; the build error is now logged with a full traceback instead of hidden.
  * **Cleanup — fewer, less redundant nodes.** Removed `Long Chunk Sampler` and `Long Chunk Sampler (Multi-Prompt)`, which were both subsumed by **`Long Chain Sampler (Dynamic)`** (one prompt per line, `num_chunks` sets the count — a single prompt + `num_chunks=N` reproduces the old single-prompt sampler, and the multiline list replaces the 6-socket multi-prompt one with no cap). Also reverted the experimental chained-timeline "lead-in ghost" UI overlay and dropped a dead `WanAnimateDirector` flag. **Note:** if you have a saved workflow using the two removed sampler nodes, swap them for `Long Chain Sampler (Dynamic)`.

**v1.14.0**
  * **`Long Video Stitcher` — dynamic latent inputs.** The latent input slots now grow as you connect chunks (and shrink when you disconnect), up to 12. Connect as many chunks as you need without pre-wiring empty slots. Audio latents still work (a second instance for the audio track).

**v1.13.0**
  * **Smoother seams — chained chunks no longer stutter.** Auto-chaining (LTX & Wan `prev_latent`) now carries a short **motion clip** (the last couple of latent frames) across the seam instead of a single still frame, so the model continues the existing motion rather than restarting from a standstill. The `2x30s Chain` example also switches the stitch to `drop` mode (no static-vs-moving crossfade). Tune `overlap_frames` on the stitcher and `_CHAIN_TAIL_LATENT_FRAMES` (in `ltx_director.py`) if a seam still needs work.
  * **New node: `Long Chain Sampler (Dynamic)`.** Set `num_chunks=N` and give one prompt per line — it samples N chunks in a single node (each seeded from the previous chunk's tail) and stitches them into one latent. The chunk count is not capped by input sockets, and `num_chunks=0` means "one chunk per prompt line". This is the fast single-stage path; for the LTX 2-stage upscale quality, chain Directors with `prev_latent` instead.

**v1.12.0**
  * **Wan Director auto-chaining — `prev_latent` input.** `Wan Director` now chains the same way LTX Director does: connect the previous chunk's output video latent to `prev_latent` and it decodes the tail frame (reusing the connected `vae`) to use as this chunk's start image, automatically resolving to an image-to-video variant (`i2v-14b` / `ti2v-5b`). An explicit timeline start-image segment still takes priority. The advanced Wan Directors (`Wan S2V` / `Wan Animate`) already chain natively via their `frame_offset` / `continue_motion` inputs.

**v1.11.0**
  * **LTX Director auto-chaining — `prev_latent` / `prev_vae` inputs.** Connect the previous chunk's output video latent to `prev_latent` (and the video VAE to `prev_vae`) and the Director automatically decodes its tail frame and uses it as this chunk's start keyframe. No more manually wiring a `Latent Tail to Image` node between chunks — just `Director → Director`. An explicit `start_image` still takes priority. The `2x30s Chain` example workflow now uses this auto-chain wiring.

**v1.10.0**
  * **Long-video chaining — smoothly stitch multiple 30s LTX clips into one**
    * **New `start_image` input on LTX Director.** Connect an image (e.g. the previous chunk's tail frame) and it is injected as a hard frame-0 keyframe, so the next chunk continues seamlessly from where the last one ended. It also defines the output canvas and skips the text-to-video dummy frame. A `start_image_strength` widget controls how hard the first frame is anchored.
    * **`Long Video Stitcher` now also stitches audio latents.** 4D audio latents (`[B,C,T,F]`) are plainly concatenated (overlap / blend ignored), so the chained chunks' audio can be combined alongside the cosine-blended video.
    * **New example workflow: `LTX Director Long Video (2x30s Chain) v1`.** Two `LTX Director → Stage #1 → Stage #2` chains bridged by `Latent Tail to Image` (chunk 1's tail → chunk 2's `start_image`), with the two chunks' video latents cosine-blended and audio latents concatenated before a single Decode → a 60s video. Extend to more chunks by duplicating the chain (up to 6 per stitcher). The pattern works around LTX having no native `frame_offset`.

**v1.4.0**
  * **Wan2.2 family support — four new Director nodes**
    * **Wan Director** — T2V-A14B / I2V-A14B (MoE 14B, two model inputs), FLF-14B, TI2V-5B. Auto-detects Wan2.1 vs Wan2.2 latent format from the model.
    * **Wan S2V Director** — Wan2.2-S2V (Speech-to-Video). Audio embedding drives lip-sync; place a reference image on the timeline for subject identity. `frame_offset` output chains long clips.
    * **Wan VACE Director** — Wan2.2-VACE. Control video (pose / depth / sketch) + optional control masks + reference image; outputs `trim_latent` to feed `TrimVideoLatent`.
    * **Wan Animate Director** — Wan2.2-Animate. Character animation with `pose_video`, `face_video`, `continue_motion` (chunk extending), `character_mask`, `background_video`.
  * All four reuse the LTX Director timeline UI for **Prompt Relay** segmented prompts. Image segments on the timeline supply the variant-specific reference / start / end images. Reference-prefix latent frames (Animate / VACE) are now correctly skipped when distributing prompt segments.

**v1.3.9**
  * **Fixed recent updates not showing in the manager**

It took like 5 tries but I finally got it working 🤦‍♂️

**v1.3.3**
  * **LTX Director Hotfix 2**
    - Fixed duration_seconds input issue.
    - Made both duration widgets visible at all times now
    - Implemented audio latent fix to improve compatibility


**v1.3.2**
  * **LTX Director Hotfix**
    - Fixed epsilon input overlapping custom_width input
    - Fixed invisible widgets in nodes 2.0 when toggling widget visibility through settings menu

If anyone finds anymore bugs or has idea for improvements please let me know! 


**v1.3.1**
  * **LTX Director Example Workflow Fix**
    - Minor fix to the example workflow (i forgot to set the clip loader type to ltxv lol)
    
 **v1.3.0**
  * **New nodes: LTX Director and LTX Director Guide**
    - A complete timeline editor that can do almost everything. It's my most ambitious node so far and the successor to LTX Sequencer/Multi Image Loader.

 **v1.2.9**
  * **Fixed every known issue with Multi Image Loader and added text output to Speech Length Calculator**
  
    - Removed the completely useless drag and drop animations (now it's snappy and no longer finicky)
    - Fixed the node resizing on nodes 2.0 
    - Updated grid logic to fit images better
    - Added ablity to right click images to copy/open/save images
    - Fixed the "invisible hitbox" underneath node issue (actually this time).

  Also added a text output to the Speech Length Calculator node (can't believe i didn't do this initially)

<details>
  <summary>Click to view older Updates</summary>

 **v1.2.8**
  * **Updated Load Video UI and Color Conversion**
    * Added crop mode, a simple interface to crop videos. It also include various aspect ratio presets.
    * Updated color conversion to ensure colors are as accurate as possible. Will first check metadata for colorspace, and if metadata is missing then it will guess the colorspace based on video dimensions.
    * Updated display mode toggle UI to be more understandable 

 **v1.2.7**
  * **New Node: Load Video UI**

Custom Node to Trim, Resize, and Preview Videos in Realtime
  
   **v1.2.6**
  * **Updated Speech Length Calculator UI**

Also added duration output to the Load Audio UI node

 **v1.2.5**
  * **Updated Load Audio UI Node**
    * Added Duration Setting
    * Made the whole selection bar draggable
    * Fixed Trimmed UI to show centiseconds
    
 **v1.2.4**
 * **New Node: Load Audio UI**

Overhaul of the load audio node. Features a simple interface to easily trim audio. Also allows dragging and dropping files (fixes the original node that doesn't allow dropping in videos). Also compatible with nodes 2.0.

 **v1.2.3**
  * **Workflow Update + Minor Bug Fix** 
    * Added new workflow that is compatible with the latest ComfyUI version (as of 4/27/26). The new workflow also included an option to include custom audio, and has minor improvements of the previous workflows.
    * Fixed minor bug with Multi Image Loader that blocked mouse input in a small area under the node 🤷‍♂️

**v1.2.0**
  * **New Node: Speech Length Calculator** 
  
  Automatically output in realtime how long a video should be based on the dialouge. 

**v1.1.0**
  * Added resize_method to the Multi Image Loader node for more resize options
  * Added insert_mode which allows you to enter in seconds instead of frames on the LTX Sequencer node
  * Updated workflows with more notes
  * Re-added tiny vae to workflows
  * Fixed various bugs
  * more things i can't rememeber
  
**This update will change the node layouts, so be sure to update your workflows or else they won't work properly.**

❗❗❗ **New Tutorial on using these nodes available: https://www.youtube.com/watch?v=aXDIr8eNovI**  ❗❗❗
</details>

# ⚙️ Custom Nodes


## Wan Director (Wan2.2 family)

A set of four Director nodes for the Alibaba **Wan2.2** video models, built on the same Prompt Relay timeline editor as LTX Director. All four share the visual timeline UI — prompt segments drive the temporal attention masking, image segments at the start / end of the timeline supply reference / start / end frames where applicable. Audio segments are ignored by the Wan nodes.

**Wan Director** — the main node, supports:
- **T2V-A14B / I2V-A14B** (Wan2.2 MoE 14B). Connect `model_high` and `model_low` to patch both halves of the MoE pair at once with the same prompt relay configuration.
- **FLF-14B** (first-last-frame). Place an image at the start of the timeline and another at the end.
- **TI2V-5B** (the 5B unified text+image to video model, uses the new 4×16×16 VAE).

Variant is auto-detected from the model's latent format and the number of image segments; can be overridden with the `model_variant` dropdown.

**Wan S2V Director** — Wan2.2-S2V (Speech-to-Video). Connect an `AudioEncoder` output for lip-sync; the first timeline image segment becomes the reference subject. Optional `control_video` and `ref_motion` inputs for spatial / motion priors. The `frame_offset` output chains into the next Director call when stitching long clips.

**Wan VACE Director** — Wan2.2-VACE. Drive generation with a `control_video` (pose / depth / sketch / canny etc.) plus optional `control_masks` and `strength`. Returns a `trim_latent` count that should be fed into a `TrimVideoLatent` node after sampling.

**Wan Animate Director** — Wan2.2-Animate (character animation). Inputs include `pose_video`, `face_video`, `reference_image`, `continue_motion` (carry forward the tail frames of a previous chunk), `character_mask`, `background_video`, `clip_vision_output`. Outputs `trim_latent`, `trim_image`, and an updated `video_frame_offset` for chunked long-video generation.

All four Director nodes accept an optional `negative` conditioning (an empty one is built from the connected CLIP if unconnected) and apply the variant-specific concat / VACE / audio conditioning to both positive and negative.


## LTX Director
<img width="1481" height="833" alt="Clipboard Image (2)" src="https://github.com/user-attachments/assets/08f3fe53-9393-4f5d-9de5-58b229fbed47" />

A Complete Timeline Editor For LTX 2.3. This is the sucessor of my previous nodes, and has loads of features in it. It was originally based off of [Kijai's Prompt Relay node](https://github.com/kijai/ComfyUI-PromptRelay) and my LTX Sequencer/Multi Image Loader nodes.

**Main Features:**
- **Fully Functional Timeline Editor:** I spent hours studying various video editors and ended up with this design. If anyone has ideas for improvements let me know! I will adding documentation on all the functions soon.
- **Prompt Relay integrated:** This unlocks the ability to have granular control over video generation. For more information on Prompt Relay go here, https://gordonchen19.github.io/Prompt-Relay/
- **First, Middle, Last Frame Support:** This has by far the easiest method of creating first/last frames videos. It supports any number of keyframes, and will be the successor of my previous nodes.
- **Custom Audio Support:** Import, trim, and combine your own audio clips in this node. Enabling custom audio is as simple as clicking 1 button. It is also compatible with every other feature in the node, include first/last frames, t2v, i2v, and prompt relay.
- **Image to Video:** Part of the goal of this node was to make it easier to do everything, including Image to Video. It has built in resize functionality, and of course all the benifits of the prompt relay and custom audio integration.
- **Text to Video:** Use text segments to create T2V videos. Compatible with all other features of the node.

Download workflows here: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI/tree/main/example_workflows

**Tutorial videos and documentation coming soon**


## Multi Image Loader
<img width="1280" height="720" alt="Multi_Image_Loader_Wide_Gif" src="https://github.com/user-attachments/assets/99b6afd8-5197-4e6c-81da-a7bd156c42c7" />

An Image loader that features a built in gallery, allowing your to easily rearrange images and output them seperately or batched together. It also combines the image resize node and LTXVPreprocess node to reduce clutter in LTX workflows.

## LTX Sequencer
![LTX_Sequencer_GIF](https://github.com/user-attachments/assets/88f27155-f50e-4cb2-b937-ab173e6bdf0b)

An overhaul of the LTXVAddGuideMulti node. It allows you to quickly create FFLF (First Frame Last Frame) videos, shot sequences, supports any number of middle frames.

Connect the Multi Image Loader node's multi_output to automatically update the node's widgets.

It also has a sync feature that syncs all LTX Sequencer nodes together in realtime, removing the need to edit every single node manually every time you want to make a change to something. 


## LTX Keyframer
<img width="1082" height="608" alt="LTX Keyframer Wide" src="https://github.com/user-attachments/assets/850ba4a2-dbca-4e5a-a580-1c271e9f0c41" />

An overhaul of the LTXVImgToVideoInplaceKJ node. It allows you to quickly create FFLF (First Frame Last Frame) videos and shot sequences. Also upports any number of middle frames.

Connect the Multi Image Loader node's multi_output to automatically update the node's widgets.

It also has a sync feature that syncs all LTX Keyframer nodes together in realtime, removing the need to edit every single node manually every time you want to make a change to something. 

**I would recommend using the LTX Sequencer Node over this node, after further testing it seems superior in at pretty much everything. I'll leave it in just in case more people want to test it**

## Speech Length Calculator
<img width="1280" height="720" alt="Speech Length Calculator v2 Gif" src="https://github.com/user-attachments/assets/04b9a1cf-20e4-4b7b-a9c6-4a5a0825995b" />
<br>
<br>
This node calculates in realtime how long a video should be based on the dialogue. Any words in quotations will be considered as speech. The node updates in realtime without having to run the workflow, and outputs the length depending on how fast the speech is.

If you connect another string/text node to the text_input, it will still update in the length in realtime.

I kept having to play the guessing game on my own generations so I made this node to make it easier :man_shrugging:

## Load Video UI  
<table width="100%">
  <tr>
    <td width="50%" align="center">
      <p>Simple Controls</p>
      <img src="https://github.com/user-attachments/assets/fb76ff03-a6ff-4837-bd63-7e429f5f3d37" width="100%" />
    </td>
    <td width="50%" align="center">
      <p>New Crop Mode!</p>
      <img src="https://github.com/user-attachments/assets/28cfb4ca-e42a-44da-9afb-f20cb01b9722" width="100%" />
    </td>
  </tr>
</table>

<br>
<br>
An upgraded Load Video node. It has the following features:

* Simple interface to quickly trim videos and preview them in realtime.
* Ability to load any length of video into the node (the default load video node was limited to 100MB files)
* Easily switch between showing seconds and frames with a toggle button. This will change the widgets as well as the interface.
* Multiple options for resizing the video (maintain aspect ratio, crop, stretch to fit, pad)
* Allows dragging and dropping files into the node
* Progress bar
* Optimized to use less RAM (still very limited due to ComfyUI limitations, but at least a little more efficient)

Please note that due to ComfyUI limitations (and the fact that this node doesn't use any addtional libraries), this node will not work well for outputting large videos. You can trim any length of video without a problem, but if the output is still large it will end up using a lot of RAM. I have implemented various optimizations though to make it use less memory.

## Load Audio UI  
<img width="1280" height="720" alt="Load_Audio_UI_V2" src="https://github.com/user-attachments/assets/e3dc5c8d-d0b9-4336-8196-944204719239" />
<br>
<br>
An upgraded Load Audio node. Features a simple interface to easily trim audio. Also allows dragging and dropping files (fixes the original node that doesn't allow dropping in videos). Also compatible with nodes 2.0.

# 💡 Workflows
Download workflows here: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI/tree/main/example_workflows

# ❗ Known Issues

Fixed everything so far. If there are any other issue or bugs you find please let me know!

# 💡 Additional Info

Feel free to suggest improvements, and if you run into any bugs let me know!

For those asking, I mainly used gemini to create these nodes.
