"""Shared data shapes passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Product:
    name: str
    features: tuple[str, ...] = ()


@dataclass
class Variant:
    """One generated ad: the strategy that produced it, its prompt, images, and scores."""
    strategy: str
    prompt: str
    images: list[Any] = field(default_factory=list)   # PIL.Image.Image; untyped so PIL is optional at import
    rating: float = 0.0        # 0 (poor) .. 1 (excellent)
    bias_skew: float = 0.0     # 0 (balanced) .. 1 (one group only)
    reward: float = 0.0
