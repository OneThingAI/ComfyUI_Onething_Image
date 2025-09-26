from .nodes import *
from .onethingai import *

# Register the node class
NODE_CLASS_MAPPINGS = {
    "OneThingAILoader": OneThingAILoader,
    "OnethingAIImageGemini":GeminiImage,
    "OnethingAIImageWanxiang":WanxiangImage,
    "OnethingAIImageHunyuan":HunyuanImage,
    "OnethingAIImageOpenAI":OpenAIImage,
    "OnethingAIImageSeedream": SeedreamImage
}

# Add descriptions
NODE_DISPLAY_NAME_MAPPINGS = {
    "OneThingAILoader": "OneThingAI Image Generator",
    "OnethingAIImageGemini": "OneThingAI Gemini Image",
    "OnethingAIImageWanxiang": "OneThingAI Wanxiang Image",
    "OnethingAIImageHunyuan": "OneThingAI Hunyuan Image",
    "OnethingAIImageOpenAI": "OneThingAI OpenAI Image",
    "OnethingAIImageSeedream": "OneThingAI Seedream Image"
}
