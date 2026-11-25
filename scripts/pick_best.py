"""Write models/best.json: the winning algorithm and run, which main.py loads.

Ranks every seed-0 sweep run in logs/sweep_summary.csv by final task return (50 eval seeds).
Also records each algorithm's best run and, where the extra seeds exist, its mean over seeds.

  python scripts/summarize_sweeps.py && python scripts/pick_best.py
  python scripts/pick_best.py --algo ppo --run 4      # pin a choice by hand
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=None)
    ap.add_argument("--run", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / "logs" / "sweep_summary.csv").open()))
    for r in rows:
        r["score"] = float(r["final_task_return"])
    base = [r for r in rows if "_seed" not in r["run_id"]
            and (ROOT / "models" / ("dqn" if r["algo"] == "dqn" else "pg")
                 / f"{r['algo']}_{r['run_id']}.pt").exists()]

    per_algo = {}
    for algo in dict.fromkeys(r["algo"] for r in base):
        top = max((r for r in base if r["algo"] == algo), key=lambda r: r["score"])
        seeds = [r["score"] for r in rows if r["algo"] == algo
                 and r["run_id"].split("_seed")[0] == top["run_id"]]
        per_algo[algo] = {"run": int(top["run"]), "run_id": top["run_id"],
                          "final_task_return": top["score"], "changes": top["changes"],
                          "seeds": len(seeds), "mean_over_seeds": round(sum(seeds) / len(seeds), 1)}

    if args.algo:
        rid = f"run{args.run:02d}"
        win = next(r for r in base if r["algo"] == args.algo and r["run_id"] == rid)
    else:
        win = max(base, key=lambda r: r["score"])
    best = {"algo": win["algo"], "run": int(win["run"]), "run_id": win["run_id"],
            "final_task_return": win["score"], "final_std": float(win["final_std"]),
            "changes": win["changes"], "chosen_by": "hand" if args.algo else "highest final task return",
            "per_algorithm": per_algo}
    out = ROOT / "models" / "best.json"
    out.write_text(json.dumps(best, indent=2))
    print(f"best: {win['algo']} {win['run_id']} ({win['score']:+.1f}) -> {out.relative_to(ROOT)}")
    for algo, b in per_algo.items():
        print(f"  {algo:10s} {b['run_id']}  {b['final_task_return']:+7.1f}"
              f"  (mean over {b['seeds']} seed{'s' if b['seeds'] > 1 else ''}: {b['mean_over_seeds']:+.1f})")


if __name__ == "__main__":
    main()
