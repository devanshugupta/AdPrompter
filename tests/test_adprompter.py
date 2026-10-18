"""Behavioural checks: the loop runs end to end offline, the reward penalises repeats and
skew, the bandit's bookkeeping is right, and the agent learns the simulated oracle."""
import json

import numpy as np
import pytest

from adprompter import (AdEnv, MockAuditor, MockGenerator, Product, SimulatedRater, STRATEGIES,
                        TemplatePromptBuilder, ThompsonAgent, excess_skew, expected_skew, run_campaign, skew)


def _env(seed=0, **kw):
    kw.setdefault("horizon", 4)
    return AdEnv(Product("Trail Runner X", ("carbon plate", "280g")), TemplatePromptBuilder(),
                 MockGenerator(size=32), SimulatedRater(seed), MockAuditor(), n_images=2, **kw)


def test_episode_runs_and_terminates_at_horizon():
    env = _env()
    obs, _ = env.reset()
    assert obs.shape == (len(STRATEGIES) + 1,)
    done, steps = False, 0
    while not done:
        _, reward, done, _, info = env.step(steps)
        assert -1.5 <= reward <= 1.0 and info["variant"].images
        steps += 1
    assert steps == 4
    with pytest.raises(RuntimeError):
        env.step(0)


def test_repeat_is_penalised():
    env = _env(repeat_penalty=0.5, bias_weight=0.0)
    env.rater.noise = 0.0
    env.reset()
    _, first, *_ = env.step(3)
    _, again, _, _, info = env.step(3)
    assert np.isclose(first - again, 0.5) and info["repeat"]


def test_horizon_cannot_exceed_the_strategy_count():
    with pytest.raises(ValueError):
        _env(horizon=len(STRATEGIES) + 1)


def test_skew_bounds_and_chance_floor():
    assert skew([1, 1, 1], 3) == 1.0 and skew([0, 1, 2], 3) == 0.0 and skew([0, 0], 1) == 0.0
    assert expected_skew(3, 2) == pytest.approx(0.5)      # three draws over two groups can never be even
    assert excess_skew([0, 0, 1], 2) == 0.0               # as balanced as three draws get: no excess
    assert excess_skew([0, 0, 0], 2) == pytest.approx(0.5)


def test_mock_auditor_is_deterministic_and_font_independent():
    gen = MockGenerator(size=32)
    a, b = gen.generate("p", 3), gen.generate("p", 3)
    assert MockAuditor().audit(a) == MockAuditor().audit(b)


def test_thompson_never_repeats_until_every_arm_is_used():
    agent = ThompsonAgent(4, seed=0)
    obs = np.zeros(5, dtype=np.float32)
    for a in (0, 1, 2):
        obs[a] = 1.0
    assert agent.select(obs) == 3
    obs[3] = 1.0
    assert agent.select(obs) in range(4)   # all used: any arm, no crash


def test_thompson_posterior_matches_closed_form():
    agent = ThompsonAgent(1, prior_mean=0.0, prior_var=1.0, noise_var=0.25)
    obs = np.zeros(2, dtype=np.float32)
    rewards = [0.8, 0.6, 1.0]
    for r in rewards:
        agent.update(obs, 0, r)
    k, s = len(rewards), sum(rewards)
    post_var = 1 / (1 / 1.0 + k / 0.25)
    assert agent.var[0] == pytest.approx(post_var)
    assert agent.mean[0] == pytest.approx(post_var * (0 / 1.0 + s / 0.25))


def test_agent_learns_the_best_strategy():
    env = _env(seed=3, bias_weight=0.0)
    agent = ThompsonAgent(len(STRATEGIES), seed=3)
    records = run_campaign(env, agent, episodes=40)
    best_true = max(env.rater.quality, key=env.rater.quality.get)
    late = records[len(records) // 2:]
    first_picks = [r.strategy for r in late if r.step == 0]   # the unconstrained choice each campaign
    assert first_picks.count(best_true) / len(first_picks) > 0.6
    assert np.mean([r.reward for r in late]) > np.mean([r.reward for r in records[: len(records) // 2]])


def test_records_and_images_are_written_in_campaign_order(tmp_path):
    env = _env(horizon=3)
    run_campaign(env, ThompsonAgent(len(STRATEGIES)), episodes=2, out_dir=tmp_path)
    records = json.loads((tmp_path / "records.json").read_text())
    assert [(r["episode"], r["step"]) for r in records] == [(e, s) for e in range(2) for s in range(3)]
    files = sorted(p.name for p in (tmp_path / "variants").glob("*.png"))
    assert len(files) == 2 * 3 * 2 and files[0].startswith("ep000_s00_") and files[-1].startswith("ep001_s02_")


def test_cli_best_strategy_ignores_unplayed(capsys, tmp_path):
    from adprompter.cli import main
    main(["--product", "P", "--episodes", "1", "--horizon", "1", "--images", "1", "--out", str(tmp_path), "--no-images"])
    out = capsys.readouterr().out
    best = out.rsplit("best strategy by mean reward: ", 1)[1].strip()
    played = {r["strategy"] for r in json.loads((tmp_path / "records.json").read_text())}
    assert best in played


def test_ppo_agent_trains_against_the_gym_adapter():
    pytest.importorskip("stable_baselines3")
    from adprompter import PPOAgent
    agent = PPOAgent(_env(), total_timesteps=64)
    assert agent.select(np.zeros(len(STRATEGIES) + 1, dtype=np.float32)) in range(len(STRATEGIES))
