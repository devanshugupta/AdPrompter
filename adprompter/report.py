"""Figures from a campaign's records: learning curve, strategy mix, and a contact sheet."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .pipeline import Record
from .strategies import STRATEGIES


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def reward_curve(records: list[Record], path: Path, window: int = 10) -> None:
    plt = _plt()
    r = np.array([x.reward for x in records])
    smooth = np.convolve(r, np.ones(window) / window, mode="valid") if len(r) >= window else r
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(r, color="#9aa5b1", lw=0.8, label="per variant")
    ax.plot(np.arange(len(smooth)) + (len(r) - len(smooth)), smooth, color="#1f5fbf", lw=2,
            label=f"{window}-variant mean")
    ax.set_xlabel("variants generated")
    ax.set_ylabel("reward")
    ax.set_title("Agent reward while learning which strategies land")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strategy_mix(records: list[Record], path: Path, last_fraction: float = 0.5) -> None:
    """Which strategies the agent chose early versus late: a learner should shift mass."""
    plt = _plt()
    n = len(records)
    early = Counter(r.strategy for r in records[: int(n * (1 - last_fraction))])
    late = Counter(r.strategy for r in records[int(n * (1 - last_fraction)):])
    idx = np.arange(len(STRATEGIES))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(idx - 0.2, [early[s] for s in STRATEGIES], width=0.4, color="#c9d3df", label=f"first {1 - last_fraction:.0%}")
    ax.bar(idx + 0.2, [late[s] for s in STRATEGIES], width=0.4, color="#1f5fbf", label=f"last {last_fraction:.0%}")
    ax.set_xticks(idx)
    ax.set_xticklabels(STRATEGIES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("times chosen")
    ax.set_title("Strategy selection shifts toward what scores")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def contact_sheet(variant_dir: Path, path: Path, cols: int = 6, limit: int = 12, thumb: int = 160) -> None:
    from PIL import Image
    files = sorted(variant_dir.glob("*.png"))[-limit:]
    if not files:
        return
    rows = -(-len(files) // cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, f in enumerate(files):
        im = Image.open(f).resize((thumb, thumb))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(path)
