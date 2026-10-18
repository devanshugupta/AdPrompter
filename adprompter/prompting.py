"""Turn (product, strategy) into an image-generation prompt.

Two builders share one interface. The template builder is deterministic and needs no
model; the LLM expander asks a fine-tuned prompter to enrich the same seed with style
tokens, the way Stable-Diffusion prompts are usually written."""
from __future__ import annotations

from typing import Protocol

from .strategies import CUES
from .types import Product


class PromptBuilder(Protocol):
    def build(self, product: Product, strategy: str) -> str: ...


class TemplatePromptBuilder:
    def build(self, product: Product, strategy: str) -> str:
        features = ", ".join(product.features) if product.features else "its key benefits"
        return (f"Advertisement photo for {product.name}, no text, {CUES[strategy]}, "
                f"highlighting {features}, professional product photography, sharp focus, 4k")


class LLMPromptExpander:
    """Expands the seed prompt with a causal LM fine-tuned on ad prompts. Imports
    transformers lazily so the package works without it."""

    def __init__(self, model_name: str = "jeong-jasonji/CSE574_prompter", max_new_tokens: int = 77,
                 device: str | None = None, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        torch.manual_seed(seed)   # sampling below is stochastic; the seed makes a run reproducible

    def build(self, product: Product, strategy: str) -> str:
        seed = f"make a convincing ad for {product.name} with themes of {strategy.lower()}"
        enc = self.tok(seed, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                                      temperature=0.7, top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True)
