"""Sanity-check the environment dynamics before any RL training.

1. Passive season (agent only WAITs): where does moisture drift, how far do crops
   grow with no care at all?
2. Random vs greedy vs greedy+fertilizer on the spec config and candidate fixes.
3. Raw environment speed (steps/second), for the compute estimate.

Usage:  python scripts/dynamics_check.py [--episodes 100]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.baselines import GreedyAgent, RandomAgent  # noqa: E402
from environment.custom_env import WAIT, ShambaGuardEnv, make_config  # noqa: E402

VARIANTS = {
    "spec_v1 (original)": make_config("spec_v1"),
    "standard (rebalanced)": make_config("standard"),
}


def passive_stats(cfg, episodes: int) -> dict:
    env = ShambaGuardEnv(cfg)
    wet = dry = inband = total = 0
    final_m, max_g, mature_day = [], [], []
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        matured_at = None
        while not done:
            _, _, term, trunc, _ = env.step(WAIT)
            done = term or trunc
            if env.step_in_day == 0:  # a day just ticked
                m = env.moisture[env.crop]
                wet += int((m > cfg.waterlog_threshold).sum())
                dry += int((m < cfg.drought_threshold).sum())
                inband += int(((m >= cfg.moisture_band[0]) & (m <= cfg.moisture_band[1])).sum())
                total += m.size
                if matured_at is None and (env.growth >= 1).any():
                    matured_at = env.day
        final_m.append(env.moisture[env._plot_mask].mean())
        max_g.append(env.growth[env._plot_mask].max())
        mature_day.append(matured_at if matured_at is not None else np.nan)
    total = max(total, 1)
    return dict(waterlogged=wet / total, drought=dry / total, in_band=inband / total,
                final_moisture=float(np.mean(final_m)), max_growth=float(np.mean(max_g)))


def agent_stats(cfg, make_agent, episodes: int) -> dict:
    env = ShambaGuardEnv(cfg)
    rows = []
    for ep in range(episodes):
        agent = make_agent(ep)
        obs, _ = env.reset(seed=10_000 + ep)
        ever_mature = np.zeros_like(env.crop)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            ever_mature |= env.growth >= 1
            done = term or trunc
        # task_return = return without potential-based shaping (comparable across configs).
        rows.append([info["task_return"], info["harvest_return"], info["plots_harvested"],
                     info["plots_lost"], info["total_yield"], info["stranded"],
                     ever_mature.sum(), info["fertilizer_used"], info["water_used"],
                     info["pesticide_used"], env.t])
    a = np.array(rows, dtype=float)
    keys = ["return", "harvest_ret", "harvested", "lost", "yield", "stranded",
            "ever_mature", "fert", "water", "pesticide", "length"]
    out = {k: a[:, i].mean() for i, k in enumerate(keys)}
    out["return_std"] = a[:, 0].std()
    return out


def speed(cfg, steps: int = 20_000) -> float:
    env = ShambaGuardEnv(cfg)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(int(rng.integers(11)))
        if term or trunc:
            env.reset()
    return steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    E = args.episodes

    print("\n== 1. Passive season (WAIT only): moisture drift and unaided growth ==")
    print(f"{'variant':<26}{'waterlog%':>10}{'drought%':>10}{'in-band%':>10}"
          f"{'final m':>9}{'max g':>7}")
    passive_variants = {**VARIANTS, "drought preset": make_config("drought")}
    for name, cfg in passive_variants.items():
        s = passive_stats(cfg, max(E // 4, 10))
        print(f"{name:<26}{100 * s['waterlogged']:>10.1f}{100 * s['drought']:>10.1f}"
              f"{100 * s['in_band']:>10.1f}{s['final_moisture']:>9.2f}{s['max_growth']:>7.2f}")

    print(f"\n== 2. Baselines ({E} episodes each; random uses {max(E // 5, 10)}) ==")
    hdr = (f"{'variant':<26}{'agent':<12}{'return':>14}{'harvest':>9}{'harv#':>7}"
           f"{'lost':>6}{'mature#':>8}{'yield':>7}{'strand%':>8}{'fert':>6}{'water':>7}")
    print(hdr)
    for name, cfg in {**VARIANTS,
                      "easy preset": make_config("easy"),
                      "drought preset": make_config("drought"),
                      "high_pest preset": make_config("high_pest")}.items():
        agents = {
            "random": (lambda ep: RandomAgent(ep), max(E // 5, 10)),
            "greedy": (lambda ep, c=cfg: GreedyAgent(c), E),
            "greedy+fert": (lambda ep, c=cfg: GreedyAgent(c, use_fertilizer=True), E),
        }
        for aname, (factory, eps) in agents.items():
            s = agent_stats(cfg, factory, eps)
            ret = f"{s['return']:.1f}±{s['return_std']:.1f}"
            print(f"{name:<26}{aname:<12}{ret:>14}{s['harvest_ret']:>9.1f}{s['harvested']:>7.1f}"
                  f"{s['lost']:>6.1f}{s['ever_mature']:>8.1f}{s['yield']:>7.2f}"
                  f"{100 * s['stranded']:>8.0f}{s['fert']:>6.1f}{s['water']:>7.1f}")
        print()

    sps = speed(make_config("standard"))
    print(f"== 3. Env speed: {sps:,.0f} steps/s (random actions, single process)")
    print(f"   200k frames of env stepping alone: ~{200_000 / sps:.0f} s")


if __name__ == "__main__":
    main()
