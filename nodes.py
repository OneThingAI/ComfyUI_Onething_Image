from .base import OpenAICompatibleNode, VolcengineNode


class GeminiImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "gemini-3-pro-image",
        "gemini-3-pro-image-2k"
        "gemini-3-pro-image-4k",
        "gemini-2.5-flash-image",
    ]
    DELETED_INPUT = [
        "image_size",
        "custom_width",
        "custom_height"
    ]
    EXTEND_INPUT = {
        "required": {
            "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                             {"default": "16:9"}),
            "seed": ("INT", {"default": "-1", "min": -1, "max": 2147483647,
                             "enabled": ["gemini-3-pro-image"]}),
        }
    }


class WanxiangImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "wanx2.1-t2i-plus",
        "wanx2.1-t2i-turbo",
        "wanx2.0-t2i-turbo",
        "wan2.5-image-preview"
    ]
    EXTEND_INPUT = {
        "optional": {
            "reference_image": ("IMAGE", {"enabled": ["wan2.5-image-preview"]})
        }
    }


class FluxImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "flux-dev"
    ]
    EXTEND_INPUT = {
        "optional": {
            "reference_image": ("IMAGE", {"enabled": []})
        }
    }


class HunyuanImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "hunyuan-image"
    ]
    EXTEND_INPUT = {
        "optional": {
            "reference_image": ("IMAGE", {"enabled": []})
        }
    }


class OpenAIImage(OpenAICompatibleNode):
    SUPPORTED_MODELS = [
        "gpt-image-1"
    ]
    EXTEND_INPUT = {
        "required": {
            "quality": (["low", "medium", "high"], {"default": "medium"}),
        }
    }


class SeedreamImage(VolcengineNode):
    SUPPORTED_MODELS = [
        "doubao-seedream-4-0-250828",
        "doubao-seedream-3-0-t2i-250415",
        "doubao-seededit-3-0-i2i-250628"
    ]
    EXTEND_INPUT = {
        "required": {
            "seed": ("INT", {"default": "-1", "min": -1, "max": 2147483647,
                             "enabled": ["doubao-seedream-3-0-t2i-250415", "doubao-seededit-3-0-i2i-250628"]}),
            "watermark": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "reference_image": ("IMAGE", {"enabled": ["doubao-seedream-4-0-250828", "doubao-seededit-3-0-i2i-250628"]}),
            "guidance_scale": ("FLOAT", {"default": 0, "min": 0, "max": 10,
                                         "enabled": ["doubao-seedream-3-0-t2i-250415",
                                                     "doubao-seededit-3-0-i2i-250628"]})
        }
    }

    def post_input(self, payload: dict, timeout: int, extra_params: dict, reference_image=None, **kwargs):
        if payload["model"] == "doubao-seededit-3-0-i2i-250628":
            payload["size"] = "adaptive"
            if reference_image is None:
                raise ValueError("doubao-seededit-3-0-i2i-250628需要参数：reference_image")
        if "guidance_scale" in kwargs and kwargs["guidance_scale"] < 1.0:
            del kwargs["guidance_scale"]
        return payload, timeout, extra_params, reference_image, kwargs
