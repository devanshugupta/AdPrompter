"""adprompter: generate ad variants and learn which persuasion strategy works.

    adprompter --product "Trail Runner X" --feature "carbon plate" --feature "280g" --episodes 20

Runs fully offline by default (mock renderer, simulated ratings). Pass --sdxl for real
images, --clip for automatic rating and bias auditing, --human to rate on the terminal,
and --ppo to train a neural policy instead of the bandit."""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent import PPOAgent, ThompsonAgent
from .bias import CLIPAuditor, MockAuditor
from .env import AdEnv
from .imaging import MockGenerator, SDXLGenerator
from .pipeline import run_campaign
from .prompting import LLMPromptExpander, TemplatePromptBuilder
from .rating import CLIPRater, HumanRater, SimulatedRater
from .report import contact_sheet, reward_curve, strategy_mix
from .strategies import STRATEGIES
from .types import Product


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adprompter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", required=True)
    p.add_argument("--feature", action="append", default=[], help="repeatable")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5, help="variants per campaign")
    p.add_argument("--images", type=int, default=3, help="images per variant")
    p.add_argument("--out", type=Path, default=Path("runs/latest"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sdxl", action="store_true", help="render with Stable Diffusion XL")
    p.add_argument("--llm-prompts", action="store_true", help="expand prompts with the fine-tuned prompter")
    p.add_argument("--clip", action="store_true", help="rate and audit with CLIP")
    p.add_argument("--human", action="store_true", help="rate on the terminal")
    p.add_argument("--ppo", action="store_true", help="train a PPO policy (stable-baselines3)")
    p.add_argument("--no-images", action="store_true", help="do not save variant images")
    a = p.parse_args(argv)

    product = Product(a.product, tuple(a.feature))
    builder = LLMPromptExpander() if a.llm_prompts else TemplatePromptBuilder()
    generator = SDXLGenerator() if a.sdxl else MockGenerator()
    rater = HumanRater() if a.human else CLIPRater() if a.clip else SimulatedRater(a.seed)
    auditor = CLIPAuditor() if a.clip else MockAuditor()
    env = AdEnv(product, builder, generator, rater, auditor, n_images=a.images, horizon=a.horizon)
    agent = PPOAgent(env, seed=a.seed) if a.ppo else ThompsonAgent(len(STRATEGIES), seed=a.seed)

    records = run_campaign(env, agent, a.episodes, a.out, keep_images=not a.no_images)
    a.out.mkdir(parents=True, exist_ok=True)
    reward_curve(records, a.out / "reward_curve.png")
    strategy_mix(records, a.out / "strategy_mix.png")
    if not a.no_images:
        contact_sheet(a.out / "variants", a.out / "contact_sheet.png")

    best = max(STRATEGIES, key=lambda s: sum(r.reward for r in records if r.strategy == s)
               / max(1, sum(1 for r in records if r.strategy == s)))
    print(f"{len(records)} variants over {a.episodes} campaigns -> {a.out}")
    print(f"mean reward first half {sum(r.reward for r in records[:len(records)//2])/max(1,len(records)//2):.3f}"
          f" -> second half {sum(r.reward for r in records[len(records)//2:])/max(1,len(records)-len(records)//2):.3f}")
    print(f"best strategy by mean reward: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
