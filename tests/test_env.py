import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (FERTILIZE, HARVEST, IRRIGATE, MOVE_E, MOVE_N, MOVE_S,
                                    MOVE_W, RECHARGE, SCAN, SPRAY, WAIT, ShambaGuardEnv,
                                    decode_obs, make_config, obs_dim)


@pytest.fixture
def env():
    # Shaping off so each test checks one action's reward on its own (shaping has its own tests).
    e = ShambaGuardEnv(make_config("standard", shaping_k=0.0))
    e.reset(seed=0)
    return e


def test_check_env():
    check_env(ShambaGuardEnv(), skip_render_check=True)


def test_observation_shape_and_range():
    env = ShambaGuardEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (261,) == (obs_dim(6),)
    assert ShambaGuardEnv(make_config("spec_v1")).observation_space.shape == (234,)
    assert obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(2000):
        obs, _, term, trunc, _ = env.step(int(rng.integers(11)))
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_same_seed_same_rollout():
    def rollout(seed):
        env = ShambaGuardEnv()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(123)
        out = [obs]
        for _ in range(300):
            obs, r, term, trunc, _ = env.step(int(rng.integers(11)))
            out.append(np.append(obs, r))
            if term or trunc:
                break
        return out

    a, b = rollout(7), rollout(7)
    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = rollout(8)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_start_state(env):
    d = decode_obs(env._get_obs(), 6)
    assert d["pos"] == (0, 0)
    assert env.battery == 100 and env.water == 20
    assert env.fertilizer == env.cfg.fertilizer_budget
    assert env.pesticide == env.cfg.pesticide_budget
    assert env.crop.sum() == 35 and not env.crop[0, 0]
    assert (env.pest > 0).sum() == 2
    assert d["known_pest"].sum() == 0  # hidden until scanned
    assert env.day == 0 and env.step_in_day == 0


def test_moves_and_edges(env):
    _, r, *_ = env.step(MOVE_N)  # edge -> invalid
    assert env.pos == (0, 0) and r == pytest.approx(-0.11)
    env.step(MOVE_S)
    assert env.pos == (1, 0)
    env.step(MOVE_E)
    assert env.pos == (1, 1)
    env.step(MOVE_W)
    env.step(MOVE_N)
    assert env.pos == (0, 0)
    assert env.battery == 100 - 4


def test_recharge_only_at_base(env):
    env.step(MOVE_E)
    _, r, *_ = env.step(RECHARGE)
    assert r == pytest.approx(-0.11)
    env.step(MOVE_W)
    env.step(RECHARGE)
    assert env.battery == 100 and env.water == 20


def test_irrigate_fertilize(env):
    env.step(MOVE_E)
    m0, n0 = env.moisture[0, 1], env.nutrient[0, 1]
    env.step(IRRIGATE)
    assert env.moisture[0, 1] == pytest.approx(min(1, m0 + 0.4))
    assert env.water == 19
    env.step(FERTILIZE)
    assert env.nutrient[0, 1] == pytest.approx(min(1, n0 + 0.3))
    assert env.fertilizer == env.cfg.fertilizer_budget - 1


def test_irrigate_on_base_is_invalid(env):
    _, r, *_ = env.step(IRRIGATE)
    assert env.water == 20 and r == pytest.approx(-0.11)


def test_scan_reveals_and_spray_clears(env):
    env.pest[:] = 0
    env.pest[1, 1] = 0.25
    env.step(MOVE_E)
    assert env._get_obs().reshape(-1)[4 * 36 + 7] == 0  # hidden
    batt = env.battery
    env.step(SCAN)
    assert env.battery == batt - 2
    assert decode_obs(env._get_obs(), 6)["known_pest"][1, 1] == pytest.approx(0.25)
    env.step(MOVE_S)  # (0,1) -> (1,1)
    _, r, *_ = env.step(SPRAY)
    assert env.pest[1, 1] == 0 and env.pesticide == env.cfg.pesticide_budget - 1
    assert r == pytest.approx(-0.01 + 0.5 - 0.2)


def test_spray_clean_penalty(env):
    env.pest[:] = 0
    env.step(MOVE_E)
    _, r, *_ = env.step(SPRAY)
    assert r == pytest.approx(-0.01 - 0.5 - 0.2)


def test_harvest_yield(env):
    env.step(MOVE_E)
    env.growth[0, 1], env.health[0, 1] = 1.0, 0.8
    _, r, *_ = env.step(HARVEST)
    assert r == pytest.approx(-0.01 + 8.0)
    assert not env.crop[0, 1]
    _, r, *_ = env.step(HARVEST)  # empty plot
    assert r == pytest.approx(-0.11)
    env.step(MOVE_E)
    env.growth[0, 2], env.health[0, 2] = 0.5, 1.0   # below harvest_min_growth: invalid
    assert not env.action_mask()[HARVEST]
    _, r, *_ = env.step(HARVEST)
    assert r == pytest.approx(-0.11)
    assert env.crop[0, 2]
    env.step(MOVE_E)
    env.growth[0, 3], env.health[0, 3] = 0.9, 1.0   # partial yield above the threshold
    _, r, *_ = env.step(HARVEST)
    assert r == pytest.approx(-0.01 + 10 * 0.3 * 0.9)


def test_action_mask_matches_invalid_actions():
    """Every masked action is rejected (r_invalid) and every unmasked one is accepted."""
    env = ShambaGuardEnv(make_config("standard"))
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        done = False
        while not done:
            mask = env.action_mask()
            a = int(rng.integers(len(mask)))
            before = env.stats["invalid_actions"]
            _, _, te, tr, info = env.step(a)
            assert (env.stats["invalid_actions"] > before) == (not mask[a])
            assert info["action_mask"].dtype == bool and info["action_mask"].shape == (len(mask),)
            done = te or tr


def test_shaping_step_formula():
    """reward = unshaped reward + gamma * phi(s') - phi(s)."""
    cfg = make_config("standard", outbreak_prob=0, start_infested=0, shaping_gamma=0.99,
                      shaping_care=0.0)
    env = ShambaGuardEnv(cfg)
    env.reset(seed=0)
    env.step(MOVE_E)
    env.growth[0, 1], env.health[0, 1] = 1.0, 0.8
    phi0 = env.potential()
    assert phi0 == pytest.approx(3.0 * 0.8)
    _, r, *_ = env.step(HARVEST)
    assert r == pytest.approx(-0.01 + 8.0 + 0.99 * env.potential() - phi0)
    for _ in range(7):   # steps 3-9 of the day; growth only changes in the day tick (step 10)
        phi0 = env.potential()
        _, r, *_ = env.step(WAIT)
        assert r == pytest.approx(-0.01 + 0.99 * env.potential() - phi0)
    phi0 = env.potential()
    _, r, *_ = env.step(WAIT)  # day tick
    assert env.potential() > phi0
    assert r == pytest.approx(-0.01 + 0.99 * env.potential() - phi0 - 3.0 * env.stats["plots_lost"])


def test_shaping_telescopes_to_zero_with_gamma_one():
    """With shaping_gamma = 1 and phi(end) = 0, shaping sums to -phi(start)."""
    env = ShambaGuardEnv(make_config("standard", shaping_gamma=1.0))
    rng = np.random.default_rng(1)
    env.reset(seed=3)
    phi_start = env.potential()
    done = False
    while not done:
        mask = env.action_mask()
        _, _, te, tr, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        done = te or tr
    assert info["shaping_return"] == pytest.approx(-phi_start, abs=1e-9)
    assert info["task_return"] == pytest.approx(info["episode_return"] + phi_start)


def test_day_tick_every_10_steps(env):
    for _ in range(9):
        env.step(WAIT)
    assert env.day == 0
    env.step(WAIT)
    assert env.day == 1 and env.step_in_day == 0


def test_terminal_at_season_end():
    """The season end is terminal (no bootstrapping past it), never truncated."""
    env = ShambaGuardEnv(make_config("standard", outbreak_prob=0, start_infested=0,
                                     overripe_damage=0, waterlog_damage=0, drought_damage=0))
    env.reset(seed=0)
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, info = env.step(WAIT)
    assert term and not trunc and info["season_end"]
    assert env.t == 450


def test_terminal_stranded(env):
    env.step(MOVE_E)
    env.battery = 1
    _, r, term, trunc, info = env.step(MOVE_E)
    assert term and not trunc and info["stranded"]
    assert r == pytest.approx(-0.01 - 20)


def test_battery_zero_at_base_is_not_stranded(env):
    env.battery = 1
    env.step(SCAN)
    assert env.battery == 0
    _, _, term, _, _ = env.step(RECHARGE)
    assert not term and env.battery == 100


def test_terminal_all_plots_done(env):
    env.crop[:] = False
    env.crop[0, 1] = True
    env.growth[0, 1] = 1.0
    env.step(MOVE_E)
    _, _, term, trunc, _ = env.step(HARVEST)
    assert term and not trunc


def test_presets_build():
    for name in ("easy", "standard", "drought", "high_pest", "spec_v1"):
        e = ShambaGuardEnv(make_config(name))
        obs, _ = e.reset(seed=0)
        assert obs.shape == (obs_dim(e.n, e.cfg.nav_features),) == e.observation_space.shape


def test_shaping_rewards_irrigating_a_dry_plot_immediately():
    env = ShambaGuardEnv(make_config("standard", outbreak_prob=0, start_infested=0))
    env.reset(seed=0)
    env.step(MOVE_E)
    env.moisture[0, 1], env.nutrient[0, 1] = 0.05, 0.6   # dry plot, nutrients in band
    _, r, *_ = env.step(IRRIGATE)
    assert r > 0.5   # vs -0.01 step cost: the care shows up on the same step

def test_nav_features_point_at_nearest_target():
    env = ShambaGuardEnv(make_config("standard", outbreak_prob=0, start_infested=0))
    env.reset(seed=0)
    env.moisture[:] = 0.5
    env.moisture[2, 3] = 0.1          # the only dry plot
    env.moisture[0, 0] = 0.0          # base cell is not a plot
    nav = decode_obs(env._get_obs(), 6)["nav"]
    assert nav["dry"]["exists"] and round(nav["dry"]["d_row"]) == 2 and round(nav["dry"]["d_col"]) == 3
    assert nav["dry"]["distance"] == pytest.approx(5 / 10)
    assert not nav["mature"]["exists"]
    assert not nav["must_return"]
    env.battery = 2
    env.pos = (1, 1)
    assert decode_obs(env._get_obs(), 6)["nav"]["must_return"]