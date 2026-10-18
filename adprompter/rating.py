"""Score a variant's images in [0, 1].

`CLIPRater` measures how well each image matches its own prompt (image-text similarity),
which is the automatic signal that replaces a human in the loop. `HumanRater` asks on the
terminal. `SimulatedRater` is the training oracle for tests and offline runs: each
strategy has a hidden true quality and ratings are that quality plus noise, so an agent
that learns should converge on the best strategy and a reward curve should rise."""
from __future__ import annotations

import random
from typing import Protocol

from .strategies import STRATEGIES
from .types import Variant

# CLIP cosine similarity for an image and a caption that match sits roughly in this band;
# it is mapped linearly onto [0, 1].
CLIP_SIM_RANGE = (0.15, 0.40)


class Rater(Protocol):
    def rate(self, variant: Variant) -> float: ...


class SimulatedRater:
    def __init__(self, seed: int = 0, noise: float = 0.08):
        rng = random.Random(seed)
        self.quality = {s: rng.uniform(0.2, 0.9) for s in STRATEGIES}
        self.noise = noise
        self.rng = random.Random(seed + 1)

    def rate(self, variant: Variant) -> float:
        q = self.quality[variant.strategy] + self.rng.gauss(0, self.noise)
        return min(1.0, max(0.0, q))


class HumanRater:
    def rate(self, variant: Variant) -> float:
        for i, img in enumerate(variant.images):
            img.show(title=f"{variant.strategy} {i + 1}")
        while True:
            raw = input(f"[{variant.strategy}] rate the {len(variant.images)} image(s) 1-10: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 10:
                return (int(raw) - 1) / 9


class CLIPRater:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self._torch = torch

    def rate(self, variant: Variant) -> float:
        inputs = self.proc(text=[variant.prompt[:300]], images=variant.images, return_tensors="pt",
                           padding=True, truncation=True)
        with self._torch.inference_mode():
            out = self.model(**inputs)
        sims = out.logits_per_image.squeeze(-1) / self.model.logit_scale.exp()   # cosine similarity
        lo, hi = CLIP_SIM_RANGE
        return float(((sims.mean() - lo) / (hi - lo)).clamp(0, 1))
