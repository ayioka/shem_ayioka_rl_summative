"""Collect finished runs into one table per algorithm (for the report's sweep tables).

Reads models/{pg,dqn}/<algo>_<run_id>.config.json (written when a run finishes) and the
sweep YAMLs (to name what each run changed). Writes logs/sweep_summary.csv and prints a
Markdown table per algorithm. All returns are task returns (no shaping): final = greedy
policy on the 50 final-eval seeds, best = best periodic eval (10 seeds) during training.

  python scripts/summarize_sweeps.py
  python scripts/summarize_sweeps.py --algos ppo dqn
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("ppo", "a2c", "reinforce", "dqn")


def run_changes(algo: str, run: int) -> str:
    spec = yaml.safe_load((ROOT / "configs" / f"{algo}.yaml").read_text())
    ch = spec["runs"].get(run) or spec["runs"].get(str(run)) or {}
    if not ch:
        return "baseline" if run == 1 else ("best combination" if run == 10 else "-")
    return ", ".join(f"{k}={v}" for k, v in ch.items())


def baselines(n_seeds: int = 50) -> list[str]:
    """Greedy and random reference scores on the same final-eval seeds (1000 ..)."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from environment.baselines import GreedyAgent
    from environment.custom_env import ShambaGuardEnv, make_config
    cfg = make_config("standard")
    env = ShambaGuardEnv(cfg)
    lines = []
    for name in ("greedy + fertilizer", "greedy", "random (valid actions)"):
        t, h, lost = [], [], []
        for s in range(1000, 1000 + n_seeds):
            agent = GreedyAgent(cfg, use_fertilizer=name.startswith("greedy +"))
            rng = np.random.default_rng(s)
            obs, info = env.reset(seed=s)
            done = False
            while not done:
                a = (int(rng.choice(np.flatnonzero(info["action_mask"]))) if name.startswith("random")
                     else agent.act(obs))
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            t.append(info["task_return"]); h.append(info["plots_harvested"]); lost.append(info["plots_lost"])
        lines.append(f"| {name} | {np.mean(t):+.1f} ± {np.std(t):.1f} | {np.mean(h):.1f} | {np.mean(lost):.1f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", choices=ALGOS, default=list(ALGOS))
    ap.add_argument("--baselines", action="store_true", help="also score greedy and random")
    args = ap.parse_args()
    if args.baselines:
        print("### Reference policies (task return, 50 final-eval seeds)\n")
        print("| Policy | Task return | Harvested | Lost |\n|---|---|---|---|")
        print("\n".join(baselines()))

    rows = []
    for algo in args.algos:
        sub = "dqn" if algo == "dqn" else "pg"
        for path in sorted((ROOT / "models" / sub).glob(f"{algo}_run*.config.json")):
            c = json.loads(path.read_text())
            m = re.match(rf"{algo}_run(\d+)(.*)\.config$", path.stem)
            run, tag = int(m.group(1)), m.group(2)
            f = c.get("final_eval", {})
            rows.append(dict(
                algo=algo, run=run, run_id=c.get("run_id", path.stem), seed=c.get("seed", 0),
                changes=run_changes(algo, run) + (f" ({tag.strip('_')})" if tag else ""),
                warm_start="yes" if (c.get("init_policy") or c.get("demo_episodes")) else "no",
                final_task_return=round(f.get("eval_task_return", float("nan")), 1),
                final_std=round(f.get("eval_task_return_std", float("nan")), 1),
                harvested=round(f.get("eval_plots_harvested", float("nan")), 1),
                lost=round(f.get("eval_plots_lost", float("nan")), 1),
                stranded=round(f.get("eval_stranded_rate", float("nan")), 2),
                best_task_return=round(c.get("best_eval", float("nan")), 1),
                minutes=round(c.get("train_time_s", 0) / 60, 1)))

    if not rows:
        print("no finished runs found under models/pg or models/dqn")
        return
    out = ROOT / "logs" / "sweep_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for algo in args.algos:
        sel = sorted((r for r in rows if r["algo"] == algo), key=lambda r: (r["seed"], r["run"]))
        if not sel:
            continue
        print(f"\n### {algo.upper()}\n")
        print("| Run | Changes | Warm start | Final task return | Harvested | Lost | Best | Min |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sel:
            print(f"| {r['run_id']} | {r['changes']} | {r['warm_start']} | "
                  f"{r['final_task_return']:+.1f} ± {r['final_std']:.1f} | {r['harvested']} | "
                  f"{r['lost']} | {r['best_task_return']:+.1f} | {r['minutes']} |")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
