"""Non-learning baselines for the report: a random agent and a greedy heuristic.

The greedy agent sees only the observation vector, like the RL agents do.
"""
from __future__ import annotations

import numpy as np

from environment.custom_env import (FERTILIZE, HARVEST, IRRIGATE, MOVE_E, MOVE_N, MOVE_S,
                                    MOVE_W, RECHARGE, SPRAY, WAIT, EnvConfig, N_ACTIONS,
                                    decode_obs)


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(N_ACTIONS))


def _step_toward(pos: tuple[int, int], target: tuple[int, int]) -> int:
    (r, c), (tr, tc) = pos, target
    if tr < r:
        return MOVE_N
    if tr > r:
        return MOVE_S
    if tc > c:
        return MOVE_E
    if tc < c:
        return MOVE_W
    return WAIT


class GreedyAgent:
    """Go to the nearest plot that is dry or visibly infested; harvest anything mature.

    Also returns to base to recharge before the battery runs out, and to refill an
    empty tank. ``use_fertilizer=True`` adds low-nutrient plots as targets.
    """

    def __init__(self, config: EnvConfig, use_fertilizer: bool = False,
                 battery_margin: int = 3, dry_threshold: float = 0.3,
                 low_nutrient: float = 0.4):
        self.cfg = config
        self.use_fertilizer = use_fertilizer
        self.margin = battery_margin
        self.dry = dry_threshold
        self.low_n = low_nutrient

    def act(self, obs: np.ndarray) -> int:
        cfg, n = self.cfg, self.cfg.grid_size
        d = decode_obs(obs, n)
        pos = d["pos"]
        r, c = pos
        battery = round(d["battery"] * cfg.battery_capacity)
        water = round(d["water"] * cfg.water_capacity)
        fert = round(d["fertilizer"] * cfg.fertilizer_budget)
        pest = round(d["pesticide"] * cfg.pesticide_budget)
        at_base = pos == (0, 0)
        crop = d["crop"]

        if at_base and (battery < 0.9 * cfg.battery_capacity or water < cfg.water_capacity):
            return RECHARGE
        if battery <= r + c + self.margin:
            return _step_toward(pos, (0, 0))

        mature = crop & (d["growth"] >= 1.0)
        dry = crop & (d["moisture"] < self.dry)
        infested = crop & (d["known_pest"] > 0)
        low_n = crop & (d["nutrient"] < self.low_n)

        # Act on the current plot first.
        if crop[r, c]:
            if mature[r, c]:
                return HARVEST
            if infested[r, c] and pest > 0:
                return SPRAY
            if dry[r, c] and water > 0:
                return IRRIGATE
            if self.use_fertilizer and low_n[r, c] and fert > 0:
                return FERTILIZE

        targets = mature.copy()
        if water > 0:
            targets |= dry
        if pest > 0:
            targets |= infested
        if self.use_fertilizer and fert > 0:
            targets |= low_n

        if targets.any():
            rows, cols = np.nonzero(targets)
            to_t = np.abs(rows - r) + np.abs(cols - c)
            reachable = battery > to_t + rows + cols + self.margin + 2
            if reachable.any():
                i = int(np.argmin(np.where(reachable, to_t, 10 ** 6)))
                return _step_toward(pos, (int(rows[i]), int(cols[i])))
            return _step_toward(pos, (0, 0))

        if water == 0 and dry.any():
            return _step_toward(pos, (0, 0))
        return WAIT


def run_episode(env, agent, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        done = terminated or truncated
    info = dict(info)
    info["terminated"] = terminated
    info["length"] = env.t
    return info
