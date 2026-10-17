"""Render a prompt into images.

`SDXLGenerator` is the real path (diffusers, GPU). `MockGenerator` draws a deterministic
placeholder from the prompt text so the whole loop, the tests, and the README figures run
on any laptop with no model download; its images carry the product name and strategy so a
contact sheet is still readable."""
from __future__ import annotations

import hashlib
from typing import Protocol

from PIL import Image, ImageDraw


class ImageGenerator(Protocol):
    def generate(self, prompt: str, n: int) -> list[Image.Image]: ...


class MockGenerator:
    def __init__(self, size: int = 256):
        self.size = size

    def generate(self, prompt: str, n: int) -> list[Image.Image]:
        return [self._draw(prompt, i) for i in range(n)]

    def _draw(self, prompt: str, i: int) -> Image.Image:
        h = hashlib.sha256(f"{prompt}|{i}".encode()).digest()
        top, bottom = tuple(h[0:3]), tuple(h[3:6])
        img = Image.new("RGB", (self.size, self.size))
        px = img.load()
        for y in range(self.size):   # vertical gradient between two prompt-derived colours
            t = y / (self.size - 1)
            row = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
            for x in range(self.size):
                px[x, y] = row
        d = ImageDraw.Draw(img)
        caption = prompt.split(",")[0].replace("Advertisement photo for ", "")[:28]
        d.rectangle([(8, self.size - 40), (self.size - 8, self.size - 8)], fill=(0, 0, 0))
        d.text((14, self.size - 34), caption, fill=(255, 255, 255))
        d.text((14, self.size - 20), f"variant {i + 1}", fill=(200, 200, 200))
        return img


class SDXLGenerator:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", steps: int = 30):
        import torch
        from diffusers import StableDiffusionXLPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps

    def generate(self, prompt: str, n: int) -> list[Image.Image]:
        return self.pipe(prompt, num_images_per_prompt=n, num_inference_steps=self.steps).images
