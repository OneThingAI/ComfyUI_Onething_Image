import base64
import io
import json
import random

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
    SUPPORTED_MODELS = []  # replace it with your own supported moldes
    DELETED_INPUT = []  # replace it with your own deleted input
    CATEGORY = "OneThingAI/image generation"
    FUNCTION = "pre_input"
    UA = "OneThingAI ComfyUI/1.1"
    URL_GENERATIONS = "https://api-model.onethingai.com/v1/images/generations"
    URL_EDIT = "https://api-model.onethingai.com/v1/images/edits"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        cls.BASE_INPUT["required"]["model"] = (cls.SUPPORTED_MODELS, {})
        required = {**cls.BASE_INPUT.get("required", {}), **cls.EXTEND_INPUT.get("required", {})}
        optional = {**cls.BASE_INPUT.get("optional", {}), **cls.EXTEND_INPUT.get("optional", {})}
        return {
            "required": {k: v for k, v in required.items() if k not in cls.DELETED_INPUT},
            "optional": {k: v for k, v in optional.items() if k not in cls.DELETED_INPUT}
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
            image_to_encode = [img for img in image]
        else:
            image_to_encode = [image]

        buffered = []
        for img in image_to_encode:
            # Convert from torch tensor to PIL Image
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            if len(img.shape) == 3:
                img = Image.fromarray(img)

            # write buffer
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            buffered.append(buf)
        print("buffered:", buffered)
        return buffered

    @staticmethod
    def read_image_result(response: requests.Response):
        response.raise_for_status()

        # Parse response
        result = response.json()

        if "data" not in result or result["data"] is None:
            raise ValueError(f"Unexpected API response: {result}")

        # Process images
        if len(result["data"]) == 0:
            raise ValueError("no image result")
        if "b64_json" not in result["data"][0] or not result["data"][0]["b64_json"]:
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

    def pre_input(self, api_key, model, prompt, retries, timeout, reference_image=None, extra=None, **kwargs):
        client = self.new_http_client(retries)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": self.UA,
        }

        payload = {
            "model": model,
            "prompt": prompt,
        }

        if "image_size" in kwargs:
            if kwargs["image_size"] == "自定义":
                payload["size"] = f"{kwargs['custom_width']}x{kwargs['custom_height']}"
            else:
                payload["size"] = kwargs["image_size"]

        if "seed" in kwargs:
            if kwargs["seed"] == -1:
                kwargs["seed"] = random.randint(1, 2147483647)

        # filter
        input_types = self.INPUT_TYPES()
        for key, settings in input_types["required"].items():
            if len(settings) > 1 and "enabled" in settings[1] and model not in settings[1]["enabled"]:
                if key in kwargs:
                    del kwargs[key]
                    # print(f"filtered input: {key} at models: {model}")
        for key, settings in input_types["optional"].items():
            if len(settings) > 1 and "enabled" in settings[1] and model not in settings[1]["enabled"]:
                if key == "reference_image":
                    reference_image = None
                    # print("reference_image off")
                    continue
                if key in kwargs:
                    del kwargs[key]
                    # print(f"filtered input: {key} at models: {model}")

        extra_params = {}

        if extra:
            try:
                extra_params = json.loads(extra)
            except json.decoder.JSONDecodeError:
                raise ValueError("extra not a json string")

        payload, timeout, extra_params, reference_image, kwargs = self.post_input(payload, timeout, extra_params,
                                                                                  reference_image, **kwargs)

        return self.generate(client, headers, payload, timeout, extra_params, reference_image, **kwargs)

    def post_input(self, payload: dict, timeout: int, extra_params: dict, reference_image=None, **kwargs) -> tuple[
        dict, int, dict, list, dict]:
        return payload, timeout, extra_params, reference_image, kwargs

    def generate(self, client: requests.Session, headers: dict[str, str], payload: dict, timeout: int,
                 extra_params: dict, reference_image=None, **kwargs):
        # Implement it
        raise NotImplementedError()


class OpenAICompatibleNode(BaseImageNode):

    def generate(self, client, headers, payload, timeout, extra_params, reference_image=None, **kwargs):
        payload.update(kwargs)
        payload.update(extra_params)
        payload["n"] = 1
        try:
            if reference_image is None:
                # generate mode, use json api
                headers["Content-Type"] = "application/json"
                response = client.post(
                    self.URL_GENERATIONS,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    verify=False
                )
                return self.read_image_result(response)

            else:
                # edit mode, use form api
                # headers["Content-Type"] = "multipart/form-data"
                img = self.read_reference_image(reference_image)
                if len(img) == 1:
                    files = {"image": img[0]}
                else:
                    files = [("image[]", i) for i in img]
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
            if len(img) == 1:
                payload["image"] = "data:image/png;base64," + base64.b64encode(img[0].getvalue()).decode()
            else:
                payload["image"] = ["data:image/png;base64,"+base64.b64encode(b64_img.getvalue()).decode() for b64_img in img]
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
