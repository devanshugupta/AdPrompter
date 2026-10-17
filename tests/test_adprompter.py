"""Behavioural checks: the loop runs end to end offline, the reward penalises repeats and
skew, and the agent learns the simulated oracle's best strategy."""
import numpy as np

from adprompter import (AdEnv, MockAuditor, MockGenerator, Product, SimulatedRater, STRATEGIES,
                        TemplatePromptBuilder, ThompsonAgent, run_campaign, skew)


def _env(seed=0, **kw):
    return AdEnv(Product("Trail Runner X", ("carbon plate", "280g")), TemplatePromptBuilder(),
                 MockGenerator(size=32), SimulatedRater(seed), MockAuditor(), n_images=2, horizon=4, **kw)


def test_episode_runs_and_terminates_at_horizon():
    env = _env()
    obs, _ = env.reset()
    assert obs.shape == (len(STRATEGIES) + 1,)
    done, steps = False, 0
    while not done:
        _, reward, done, _, info = env.step(steps % len(STRATEGIES))
        assert -1.5 <= reward <= 1.0 and info["variant"].images
        steps += 1
    assert steps == 4


def test_repeat_is_penalised():
    env = _env(repeat_penalty=0.5, bias_weight=0.0)
    env.rater.noise = 0.0
    env.reset()
    _, first, *_ = env.step(3)
    _, again, *_ = env.step(3)
    assert np.isclose(first - again, 0.5)


def test_skew_bounds():
    assert skew([1, 1, 1], 3) == 1.0 and skew([0, 1, 2], 3) == 0.0


def test_agent_learns_the_best_strategy():
    env = _env(seed=3)
    agent = ThompsonAgent(len(STRATEGIES), seed=3)
    records = run_campaign(env, agent, episodes=40)
    best_true = max(env.rater.quality, key=env.rater.quality.get)
    late = records[len(records) // 2:]
    first_picks = [r.strategy for r in late if r.step == 0]   # the unconstrained choice each campaign
    assert first_picks.count(best_true) / len(first_picks) > 0.6
    early_mean = np.mean([r.reward for r in records[: len(records) // 2]])
    assert np.mean([r.reward for r in late]) > early_mean
