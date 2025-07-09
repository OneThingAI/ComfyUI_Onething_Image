import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from PIL import Image
import io
import base64
import numpy as np
import torch

class OneThingAILoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "gpt4o"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "retries": ("INT", {"default": 3, "min": 0, "max": 5}),
                "timeout": ("INT", {"default": 20, "min": 5, "max": 100}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "generate"
    CATEGORY = "image generation"

    def generate(self, api_key, prompt, model="gpt4o", num_images=1, width=1024, height=1024, retries=3, timeout=8):
        # API endpoint
        url = "https://api-model.onethingai.com/v1/images/generations"
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Create session with retry strategy
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Request headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "n": num_images,
            "size": f"{width}x{height}",
        }
            
        try:
            # Make API request with timeout
            response = session.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if "data" not in result:
                raise ValueError(f"Unexpected API response: {result}")
            
            # Process images
            images = []
            for img_data in result["data"]:
                if "b64_json" in img_data:
                    # Decode base64 image
                    image_bytes = base64.b64decode(img_data["b64_json"])
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Convert to numpy array
                    image_array = np.array(image)
                    images.append(image_array)
                    
            # Stack images if multiple were generated
            if len(images) > 1:
                final_image = np.stack(images)
            else:
                final_image = images[0]
            
            # Convert numpy array to torch tensor
            final_image = torch.from_numpy(final_image).float()/255.0
            final_image = final_image.unsqueeze(0)
            
            return (final_image,)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse API response")
        except Exception as e:
            raise RuntimeError(f"Error generating images: {str(e)}")
        finally:
            session.close()

# Register the node class
NODE_CLASS_MAPPINGS = {
    "OneThingAILoader": OneThingAILoader
}

# Add descriptions
NODE_DISPLAY_NAME_MAPPINGS = {
    "OneThingAILoader": "OneThingAI Image Generator"
} 