"""Measure demographic skew across a variant's images.

An ad that only ever depicts one group is a liability, so skew is a cost in the reward.
`skew` is the total-variation distance between the observed attribute distribution and
uniform, scaled to [0, 1]. A handful of images cannot be exactly uniform, so a few draws
from a perfectly fair generator still score above zero; `excess_skew` subtracts that
chance level (the expected skew of n uniform draws), and the auditors report the worst
excess over attributes. `CLIPAuditor` labels each image zero-shot; `MockAuditor` derives a
stable pseudo-label from the prompt the mock renderer stamps on each image."""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from itertools import product
from typing import Protocol, Sequence

ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "gender": ("a photo of a man", "a photo of a woman"),
    "age": ("a photo of a young person", "a photo of a middle-aged person", "a photo of an elderly person"),
    "skin tone": ("a photo of a person with light skin", "a photo of a person with medium skin",
                  "a photo of a person with dark skin"),
}


def skew(labels: Sequence[int], n_groups: int) -> float:
    """0 when every group appears equally, 1 when a single group appears every time."""
    if not labels or n_groups < 2:
        return 0.0
    counts = Counter(labels)
    uniform = 1.0 / n_groups
    return 0.5 * sum(abs(counts.get(g, 0) / len(labels) - uniform) for g in range(n_groups)) / (1 - uniform)


@lru_cache(maxsize=None)
def expected_skew(n: int, n_groups: int) -> float:
    """Mean skew of n labels drawn uniformly from n_groups: the floor a fair generator hits."""
    if n == 0 or n_groups < 2:
        return 0.0
    draws = list(product(range(n_groups), repeat=n))
    return sum(skew(d, n_groups) for d in draws) / len(draws)


def excess_skew(labels: Sequence[int], n_groups: int) -> float:
    return max(0.0, skew(labels, n_groups) - expected_skew(len(labels), n_groups))


class BiasAuditor(Protocol):
    def audit(self, images) -> float: ...


def _worst(label_fn, images) -> float:
    return max((excess_skew(label_fn(images, groups), len(groups)) for groups in ATTRIBUTES.values()), default=0.0)


class MockAuditor:
    def audit(self, images) -> float:
        def labels(ims, groups):
            return [hashlib.sha256(f"{im.info.get('prompt', '')}|{im.info.get('index', i)}|{groups[0]}".encode())
                    .digest()[0] % len(groups) for i, im in enumerate(ims)]
        return _worst(labels, images)


class CLIPAuditor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self._torch = torch

    def audit(self, images) -> float:
        def labels(ims, groups):
            inputs = self.proc(text=list(groups), images=ims, return_tensors="pt", padding=True)
            with self._torch.inference_mode():
                return self.model(**inputs).logits_per_image.argmax(-1).tolist()
        return _worst(labels, images)
