"""Policies that choose the next strategy.

`ThompsonAgent` is the default: a Gaussian bandit over strategies that samples a plausible
quality for each arm from its posterior and plays the best, which trades exploration for
exploitation without any tuning. It masks arms the observation marks as already used, so
it never pays the repeat penalty. `PPOAgent` is the neural alternative through
stable-baselines3 for when the reward depends on the full campaign state; it trains on the
live environment, so under a real renderer every training step costs a render."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import AdEnv, gym_adapter, used_mask


class Agent(Protocol):
    def select(self, obs: np.ndarray) -> int: ...
    def update(self, obs: np.ndarray, action: int, reward: float) -> None: ...


class ThompsonAgent:
    def __init__(self, n_actions: int, seed: int = 0, prior_mean: float = 0.5, prior_var: float = 1.0,
                 noise_var: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.mean = np.full(n_actions, prior_mean)
        self.var = np.full(n_actions, prior_var)
        self.noise_var = noise_var

    def select(self, obs: np.ndarray) -> int:
        sample = self.rng.normal(self.mean, np.sqrt(self.var))
        used = used_mask(obs)
        if not used.all():
            sample[used] = -np.inf
        return int(sample.argmax())

    def update(self, obs: np.ndarray, action: int, reward: float) -> None:
        # Conjugate Gaussian update: precisions add, the mean is precision-weighted.
        precision = 1 / self.var[action] + 1 / self.noise_var
        self.mean[action] = (self.mean[action] / self.var[action] + reward / self.noise_var) / precision
        self.var[action] = 1 / precision


class PPOAgent:
    def __init__(self, env: AdEnv, total_timesteps: int = 2_000, seed: int = 0):
        from stable_baselines3 import PPO
        self.model = PPO("MlpPolicy", gym_adapter(env), seed=seed, verbose=0, n_steps=64, batch_size=32)
        self.model.learn(total_timesteps=total_timesteps)

    def select(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def update(self, obs, action, reward) -> None:
        pass   # PPO learns during .learn(); online updates are not part of its contract
