"""The campaign as a decision process.

One episode designs a short campaign: each step the agent picks a persuasion strategy, the
pipeline renders and scores that variant, and the reward is its rating minus a penalty for
repeating a strategy already used this episode and a penalty for demographic skew. The
observation is which strategies have been used plus how far along the episode is, which
is everything the policy needs to avoid repeats and pace itself.

The interface mirrors gymnasium (`reset() -> obs, info`, `step(a) -> obs, reward,
terminated, truncated, info`) without importing it, so the environment runs with numpy
alone and also drops into stable-baselines3 through `GymAdapter`."""
from __future__ import annotations

import numpy as np

from .bias import BiasAuditor
from .imaging import ImageGenerator
from .prompting import PromptBuilder
from .rating import Rater
from .strategies import STRATEGIES
from .types import Product, Variant


class AdEnv:
    def __init__(self, product: Product, builder: PromptBuilder, generator: ImageGenerator,
                 rater: Rater, auditor: BiasAuditor, *, n_images: int = 3, horizon: int = 5,
                 repeat_penalty: float = 0.5, bias_weight: float = 0.5):
        self.product, self.builder, self.generator = product, builder, generator
        self.rater, self.auditor = rater, auditor
        self.n_images, self.horizon = n_images, horizon
        self.repeat_penalty, self.bias_weight = repeat_penalty, bias_weight
        self.n_actions = len(STRATEGIES)
        self.obs_dim = self.n_actions + 1
        self.used: set[int] = set()
        self.t = 0

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[list(self.used)] = 1.0
        obs[-1] = self.t / self.horizon
        return obs

    def reset(self, seed: int | None = None):
        self.used.clear()
        self.t = 0
        return self._obs(), {}

    def step(self, action: int):
        strategy = STRATEGIES[action]
        v = Variant(strategy=strategy, prompt=self.builder.build(self.product, strategy))
        v.images = self.generator.generate(v.prompt, self.n_images)
        v.rating = self.rater.rate(v)
        v.bias_skew = self.auditor.audit(v.images)
        v.reward = v.rating - self.bias_weight * v.bias_skew - (self.repeat_penalty if action in self.used else 0.0)
        self.used.add(action)
        self.t += 1
        return self._obs(), v.reward, self.t >= self.horizon, False, {"variant": v}


class GymAdapter:
    """Wraps AdEnv as a gymnasium.Env for stable-baselines3; imported only when used."""

    def __init__(self, env: AdEnv):
        import gymnasium as gym
        self.env = env
        self.action_space = gym.spaces.Discrete(env.n_actions)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(env.obs_dim,), dtype=np.float32)
        self.metadata, self.render_mode, self.spec = {}, None, None

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed)

    def step(self, action):
        return self.env.step(int(action))

    def close(self):
        pass
