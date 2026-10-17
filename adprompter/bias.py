"""Measure demographic skew across a variant's images.

An ad that only ever depicts one group is a liability, so skew is a cost in the reward.
`skew` is the total-variation distance between the observed attribute distribution and
uniform: 0 when groups appear equally, 1 when a single group appears every time.
`CLIPAuditor` labels each image zero-shot; `MockAuditor` derives a stable pseudo-label
from the image bytes so the pipeline runs without a model."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Protocol, Sequence

ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "gender": ("a photo of a man", "a photo of a woman"),
    "age": ("a photo of a young person", "a photo of a middle-aged person", "a photo of an elderly person"),
    "skin tone": ("a photo of a person with light skin", "a photo of a person with medium skin",
                  "a photo of a person with dark skin"),
}


def skew(labels: Sequence[int], n_groups: int) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    uniform = 1.0 / n_groups
    return 0.5 * sum(abs(counts.get(g, 0) / len(labels) - uniform) for g in range(n_groups)) / (1 - uniform)


class BiasAuditor(Protocol):
    def audit(self, images) -> float: ...


class MockAuditor:
    def audit(self, images) -> float:
        worst = 0.0
        for groups in ATTRIBUTES.values():
            labels = [hashlib.sha256(im.tobytes()).digest()[0] % len(groups) for im in images]
            worst = max(worst, skew(labels, len(groups)))
        return worst


class CLIPAuditor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self._torch = torch

    def audit(self, images) -> float:
        worst = 0.0
        for groups in ATTRIBUTES.values():
            inputs = self.proc(text=list(groups), images=images, return_tensors="pt", padding=True)
            with self._torch.inference_mode():
                labels = self.model(**inputs).logits_per_image.argmax(-1).tolist()
            worst = max(worst, skew(labels, len(groups)))
        return worst


if __name__ == "__main__":
    assert skew([0, 0, 0, 0], 2) == 1.0
    assert skew([0, 1, 0, 1], 2) == 0.0
    assert skew([0, 1, 2], 3) == 0.0
    assert 0 < skew([0, 0, 1], 2) < 1
    print("ok")
