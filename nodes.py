from .base import OpenAICompatibleNode, VolcengineNode


class GeminiImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = ["gemini-2.5-flash-image"]

class WanxiangImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "wanx2.1-t2i-plus",
        "wanx2.1-t2i-turbo",
        "wanx2.0-t2i-turbo",
        "flux-dev",
    ]

class HunyuanImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "hunyuan-image"
    ]

class OpenAIImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "gpt-image-1"
    ]

class SeedreamImage(VolcengineNode):
    SUPPORTED_MODELS = [
        "doubao-seedream-4-0-250828",
        "doubao-seedream-3-0-t2i-250415",
        "doubao-seededit-3-0-i2i-250628"
    ]
    EXTEND_INPUT = {
        "required": {
            "seed": ("INT", {"default": "-1", "min": -1, "max": 2147483647}),
            "watermark": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "guidance_scale": ("FLOAT", {"min": 1, "max": 10})
        }
    }