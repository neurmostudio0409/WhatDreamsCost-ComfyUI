from .ltx_keyframer import LTXKeyframer
from .multi_image_loader import MultiImageLoader
from .ltx_sequencer import LTXSequencer
from .speech_length_calculator import SpeechLengthCalculator
from .load_audio_ui import LoadAudioUI
from .load_video_ui import LoadVideoUI
from .ltx_director import LTXDirector
from .ltx_director_guide import LTXDirectorGuide
from .wan_director import WanDirector, WanS2VDirector, WanVaceDirector, WanAnimateDirector
from .long_video_stitcher import (
    LongVideoStitcher,
    LatentTailToImage,
    LongChunkSampler,
    LongChunkSamplerMulti,
    LightningLoraPreset,
)
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXDirector,
            LTXDirectorGuide,
            WanDirector,
            WanS2VDirector,
            WanVaceDirector,
            WanAnimateDirector,
            LongVideoStitcher,
            LatentTailToImage,
            LongChunkSampler,
            LongChunkSamplerMulti,
            LightningLoraPreset,
        ]

async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "LTXKeyframer": LTXKeyframer,
    "MultiImageLoader": MultiImageLoader,
    "LTXSequencer": LTXSequencer,
    "SpeechLengthCalculator": SpeechLengthCalculator,
    "LoadAudioUI": LoadAudioUI,
    "LoadVideoUI": LoadVideoUI,
    "LTXDirector": LTXDirector,
    "LTXDirectorGuide": LTXDirectorGuide,
    "WanDirector": WanDirector,
    "WanS2VDirector": WanS2VDirector,
    "WanVaceDirector": WanVaceDirector,
    "WanAnimateDirector": WanAnimateDirector,
    "LongVideoStitcher": LongVideoStitcher,
    "LatentTailToImage": LatentTailToImage,
    "LongChunkSampler": LongChunkSampler,
    "LongChunkSamplerMulti": LongChunkSamplerMulti,
    "LightningLoraPreset": LightningLoraPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXKeyframer": "LTX Keyframer",
    "MultiImageLoader": "Multi Image Loader",
    "LTXSequencer": "LTX Sequencer",
    "SpeechLengthCalculator": "Speech Length Calculator",
    "LoadAudioUI": "Load Audio UI",
    "LoadVideoUI": "Load Video UI",
    "LTXDirector": "LTX Director",
    "LTXDirectorGuide": "LTX Director Guide",
    "WanDirector": "Wan Director",
    "WanS2VDirector": "Wan S2V Director",
    "WanVaceDirector": "Wan VACE Director",
    "WanAnimateDirector": "Wan Animate Director",
    "LongVideoStitcher": "Long Video Stitcher",
    "LatentTailToImage": "Latent Tail to Image",
    "LongChunkSampler": "Long Chunk Sampler",
    "LongChunkSamplerMulti": "Long Chunk Sampler (Multi-Prompt)",
    "LightningLoraPreset": "Lightning LoRA Preset",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']