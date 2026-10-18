"""Run campaigns and keep what they produced."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import Agent
from .env import AdEnv
from .types import Variant


@dataclass
class Record:
    episode: int
    step: int
    strategy: str
    rating: float
    bias_skew: float
    reward: float
    prompt: str


def run_campaign(env: AdEnv, agent: Agent, episodes: int, out_dir: Path | None = None,
                 keep_images: bool = True) -> list[Record]:
    """Play `episodes` campaigns, updating the agent after every variant. Returns one
    record per variant. With out_dir, records.json is rewritten after every variant (a
    crashed run keeps what it produced) and images land under out_dir/variants."""
    records: list[Record] = []
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.select(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.update(obs, action, reward)
            v: Variant = info["variant"]
            records.append(Record(ep, step, v.strategy, v.rating, v.bias_skew, v.reward, v.prompt))
            if out_dir:
                if keep_images:
                    _save_images(v, out_dir / "variants", ep, step)
                (out_dir / "records.json").write_text(json.dumps([asdict(r) for r in records], indent=1))
            obs, step = next_obs, step + 1
    return records


def _save_images(v: Variant, folder: Path, ep: int, step: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    slug = v.strategy.lower().replace(" ", "-")
    for i, img in enumerate(v.images):   # zero-padded so lexical order is campaign order
        img.save(folder / f"ep{ep:03d}_s{step:02d}_{slug}_{i}.png")
