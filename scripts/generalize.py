"""Generalization test (spec Fig 6): trained agents vs baselines on conditions never trained on.

Conditions (50 episodes each):
  train seeds   standard preset, seeds 1000-1049 (the seeds the final eval already used)
  unseen seeds  standard preset, seeds 5000-5049 (never used in training or evaluation)
  drought       'drought' preset (rain much rarer, evaporation x1.5), seeds 5000-5049
  high pest     'high_pest' preset (outbreak rate x3), seeds 5000-5049

Agents: each algorithm's best run (every seed of it found in models/, e.g. run04,
run04_seed1, run04_seed2), plus greedy, greedy+fert and random. All are scored on
task_return (shaping excluded). Writes logs/generalization.csv (one row per agent model x
condition) and prints a table.

  python scripts/generalize.py                         # best runs read from logs/sweep_summary.csv
  python scripts/generalize.py --best ppo=4 dqn=9      # choose runs by hand
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.baselines import GreedyAgent  # noqa: E402
from environment.custom_env import ShambaGuardEnv, make_config  # noqa: E402
from training.policy import Policy, model_path  # noqa: E402

ALGOS = ("ppo", "a2c", "reinforce", "dqn")
CONDITIONS = {  # name -> (preset, first seed)
    "train seeds": ("standard", 1000),
    "unseen seeds": ("standard", 5000),
    "drought": ("drought", 5000),
    "high pest": ("high_pest", 5000),
}
METRICS = ("task_return", "plots_harvested", "plots_lost", "total_yield", "stranded")


def best_runs_from_summary() -> dict[str, int]:
    """Highest final task return among seed-0 runs, per algorithm."""
    path = ROOT / "logs" / "sweep_summary.csv"
    best: dict[str, tuple[float, int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["seed"] not in ("0", "") or "_seed" in row["run_id"]:
                continue
            score = float(row["final_task_return"])
            if row["algo"] not in best or score > best[row["algo"]][0]:
                best[row["algo"]] = (score, int(row["run"]))
    return {a: r for a, (_, r) in best.items()}


class NetAgent:
    """A trained policy (training/policy.py), acting greedily over the valid actions."""

    def __init__(self, policy: Policy):
        self.policy, self.mask = policy, None

    def act(self, obs):
        return self.policy.predict(obs, self.mask)


def play(agent, preset: str, seeds: list[int]) -> dict[str, np.ndarray]:
    env = ShambaGuardEnv(make_config(preset))
    out = {m: [] for m in METRICS}
    for s in seeds:
        obs, info = env.reset(seed=s)
        done = False
        while not done:
            if isinstance(agent, NetAgent):
                agent.mask = info["action_mask"]
            obs, _, term, trunc, info = env.step(agent.act(obs))
            done = term or trunc
        for m in METRICS:
            out[m].append(float(info[m]))
    return {m: np.array(v) for m, v in out.items()}


def play_random(preset: str, seeds: list[int]) -> dict[str, np.ndarray]:
    """Uniform over the currently valid actions."""
    env = ShambaGuardEnv(make_config(preset))
    out = {m: [] for m in METRICS}
    for s in seeds:
        rng = np.random.default_rng(s)
        env.reset(seed=s)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(env.action_mask()))))
            done = term or trunc
        for m in METRICS:
            out[m].append(float(info[m]))
    return {m: np.array(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", nargs="*", default=[], metavar="ALGO=RUN")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="logs/generalization.csv")
    args = ap.parse_args()

    best = best_runs_from_summary()
    best.update({k: int(v) for k, v in (item.split("=") for item in args.best)})
    print("best runs:", best)

    agents: list[tuple[str, str, str, object]] = []  # (agent label, algo, model id, agent)
    for algo in ALGOS:
        if algo not in best:
            continue
        base = f"run{best[algo]:02d}"
        for rid in (base, f"{base}_seed1", f"{base}_seed2"):
            if model_path(algo, rid).exists():
                agents.append((f"{algo.upper() if algo != 'reinforce' else 'REINFORCE'}", algo, rid,
                               NetAgent(Policy(algo, rid))))
    agents += [("Greedy + fert", "baseline", "greedy+fert", "greedy+fert"),
               ("Greedy", "baseline", "greedy", "greedy"),
               ("Random", "baseline", "random", "random")]

    rows = []
    for cond, (preset, seed0) in CONDITIONS.items():
        seeds = list(range(seed0, seed0 + args.episodes))
        cfg = make_config(preset)
        for label, algo, rid, agent in agents:
            if agent == "random":
                res = play_random(preset, seeds)
            elif agent in ("greedy", "greedy+fert"):
                res = play(GreedyAgent(cfg, use_fertilizer=agent == "greedy+fert"), preset, seeds)
            else:
                res = play(agent, preset, seeds)
            row = {"condition": cond, "preset": preset, "agent": label, "algo": algo, "model": rid,
                   "episodes": len(seeds)}
            for m in METRICS:
                row[m] = round(float(res[m].mean()), 2)
            row["task_return_std"] = round(float(res["task_return"].std()), 2)
            rows.append(row)
            print(f"{cond:13s} {label:14s} {rid:12s} task {row['task_return']:+7.1f} ± "
                  f"{row['task_return_std']:5.1f} | harvested {row['plots_harvested']:4.1f} "
                  f"lost {row['plots_lost']:4.1f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out.with_suffix(".json")).write_text(json.dumps({"best_runs": best}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
