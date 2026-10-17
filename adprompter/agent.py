"""Policies that choose the next strategy.

`ThompsonAgent` is the default: a Gaussian bandit over strategies that samples a plausible
quality for each arm from its posterior and plays the best, which trades exploration for
exploitation without any tuning. It masks arms the observation marks as already used, so
it never pays the repeat penalty. `PPOAgent` is the neural alternative through
stable-baselines3 for when the reward depends on the full campaign state."""
from __future__ import annotations

from typing import Protocol

import numpy as np


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
        used = obs[:-1] > 0.5
        if not used.all():
            sample[used] = -np.inf
        return int(sample.argmax())

    def update(self, obs: np.ndarray, action: int, reward: float) -> None:
        # Conjugate Gaussian update: precision adds, mean is precision-weighted.
        precision = 1 / self.var[action] + 1 / self.noise_var
        self.mean[action] = (self.mean[action] / self.var[action] + reward / self.noise_var) / precision
        self.var[action] = 1 / precision


class PPOAgent:
    def __init__(self, env, total_timesteps: int = 2_000, seed: int = 0):
        from stable_baselines3 import PPO
        from .env import GymAdapter
        self.model = PPO("MlpPolicy", GymAdapter(env), seed=seed, verbose=0, n_steps=64, batch_size=32)
        self.model.learn(total_timesteps=total_timesteps)

    def select(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def update(self, obs, action, reward) -> None:
        pass   # PPO learns during .learn(); online updates are not part of its contract


if __name__ == "__main__":
    # The agent must find the best arm of a noisy bandit.
    rng = np.random.default_rng(0)
    truth = np.array([0.2, 0.9, 0.4])
    agent = ThompsonAgent(3, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    picks = []
    for _ in range(300):
        a = agent.select(obs)
        agent.update(obs, a, truth[a] + rng.normal(0, 0.05))
        picks.append(a)
    assert np.bincount(picks[-100:], minlength=3).argmax() == 1, picks[-20:]
    print("ok")
