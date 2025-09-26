import base64
import io
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import numpy as np
import torch


# Base ComfyUI defined nodes
class BaseImageNode:
    BASE_INPUT = {
        "required": {
            "api_key": ("STRING", {"default": "", "multiline": False}),
            "model": ([], {}),
            "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "提示词"}),
            "image_size": (["1536x1024", "1024x1024", "1024x1536", "自定义"], {"default": "1024x1024"}),
            "custom_width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
            "custom_height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
            "retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            "timeout": ("INT", {"default": 120, "min": 5, "max": 180}),
        },
        "optional": {
            "reference_image": ("IMAGE",),
            "extra": ("STRING",)
        }
    }
    EXTEND_INPUT = {}  # replace it with your own input to extend
    SUPPORTED_MODELS = [] # replace it with your own supported moldes
    CATEGORY = "OneThingAI/image generation"
    FUNCTION = "pre_input"
    UA = "OneThingAI ComfyUI/1.0"
    URL_GENERATIONS = "https://api-model.onethingai.com/v1/images/generations"
    URL_EDIT = "https://api-model.onethingai.com/v1/images/edits"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        cls.BASE_INPUT["required"]["model"] = (cls.SUPPORTED_MODELS, {})
        return {
            "required": {**cls.BASE_INPUT.get("required", {}), **cls.EXTEND_INPUT.get("required", {})},
            "optional": {**cls.BASE_INPUT.get("optional", {}), **cls.EXTEND_INPUT.get("optional", {})}
        }

    @staticmethod
    def new_http_client(retries) -> requests.Session:
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def read_reference_image(image):
        # Convert the first image if it's a batch
        if len(image.shape) == 4:
            image_to_encode = image[0]
        else:
            image_to_encode = image

        # Convert from torch tensor to PIL Image
        image_to_encode = (image_to_encode.cpu().numpy() * 255).astype(np.uint8)
        if len(image_to_encode.shape) == 3:
            image_to_encode = Image.fromarray(image_to_encode)

        # write buffer
        buffered = io.BytesIO()
        image_to_encode.save(buffered, format="PNG")
        return buffered

    @staticmethod
    def read_image_result(response: requests.Response):
        response.raise_for_status()

        # Parse response
        result = response.json()

        if "data" not in result:
            raise ValueError(f"Unexpected API response: {result}")

        # Process images
        if len(result["data"]) == 0:
            raise ValueError("no image result")
        if "b64_json" not in result["data"][0]:
            raise ValueError("no b64 json in result")
        image_bytes = base64.b64decode(result["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        final_image = np.array(image)
        final_image = torch.from_numpy(final_image).float() / 255.0
        final_image = final_image.unsqueeze(0)
        return (final_image,)

    def pre_input(self, api_key, model, prompt, image_size, custom_width, custom_height, retries, timeout,
                  reference_image=None, extra=None, **kwargs):
        client = self.new_http_client(retries)
        if image_size == "自定义":
            image_size = f"{custom_width}x{custom_height}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": self.UA,
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "size": image_size
        }

        extra_params = {}

        if extra:
            try:
                extra_params = json.loads(extra)
            except json.decoder.JSONDecodeError:
                raise ValueError("extra not a json string")

        return self.generate(client, headers, payload, timeout, extra_params, reference_image, **kwargs)

    def generate(self, client: requests.Session, headers: dict[str, str], payload: dict, timeout: int,
                 extra_params: dict, reference_image=None, **kwargs):
        # Implement it
        raise NotImplementedError()


class OpenAICompatibleNode(BaseImageNode):

    def generate(self, client, headers, payload, timeout, extra_params, reference_image=None, **kwargs):
        payload.update(kwargs)
        payload.update(extra_params)
        try:
            if reference_image is None:
                # generate mode, use json api
                headers["Content-Type"] = "application/json"
                response = client.post(
                    self.URL_GENERATIONS,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                return self.read_image_result(response)

            else:
                # edit mode, use form api
                # headers["Content-Type"] = "multipart/form-data"
                img = self.read_reference_image(reference_image)
                files = {"image[]": img}
                response = client.post(
                    self.URL_EDIT,
                    headers=headers,
                    data=payload,
                    files=files,
                    timeout=timeout,
                    verify=False
                )
                return self.read_image_result(response)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse API response")
        except Exception as e:
            raise RuntimeError(f"Error generating images: {str(e)}")
        finally:
            client.close()


class VolcengineNode(BaseImageNode):

    def generate(self, client, headers, payload, timeout, extra_params, reference_image=None, **kwargs):
        payload.update(kwargs)
        payload.update(extra_params)
        headers["Content-Type"] = "application/json"
        if reference_image is not None:
            img = self.read_reference_image(reference_image)
            payload["image"] = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
        try:
            response = client.post(
                self.URL_GENERATIONS,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=False
            )
            return self.read_image_result(response)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse API response")
        except Exception as e:
            raise RuntimeError(f"Error generating images: {str(e)}")
        finally:
            client.close()
