"""Run hyperparameter sweeps as parallel worker processes; resumable.

A run counts as done once its models/<dir>/<algo>_<run_id>.config.json exists, so re-running
the same command (e.g. after a Colab disconnect) only starts the runs that are missing.
Unfinished runs restart from scratch.

Examples:
  python scripts/run_sweep.py --algos ppo --workers 2                 # PPO runs 0-10
  python scripts/run_sweep.py --algos dqn a2c --runs 0 1 2 --workers 2
  python scripts/run_sweep.py --algos ppo --runs 1 --seeds 1 2        # extra seeds -> run01_seed1 ...
  python scripts/run_sweep.py --algos ppo --total-frames 4096 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("ppo", "a2c", "reinforce", "dqn")


def run_id(run: int, seed: int) -> str:
    return f"run{run:02d}" + (f"_seed{seed}" if seed else "")


def done_marker(algo: str, rid: str) -> Path:
    sub = "dqn" if algo == "dqn" else "pg"
    return ROOT / "models" / sub / f"{algo}_{rid}.config.json"


def is_current(marker: Path, expected: dict) -> bool:
    """A finished run counts only if its saved settings still match the YAML (so editing
    a run, e.g. filling in run 10, makes it run again)."""
    import json
    saved = json.loads(marker.read_text())
    return all(saved.get(k) == v for k, v in expected.items())


def command(algo: str, run: int, seed: int, extra: list[str]) -> list[str]:
    script = "training/dqn_training.py" if algo == "dqn" else "training/pg_training.py"
    cmd = [sys.executable, script, "--run", str(run), "--seed", str(seed)]
    if algo != "dqn":
        cmd += ["--algo", algo]
    if seed:
        cmd += ["--tag", f"_seed{seed}"]
    return cmd + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", choices=ALGOS, default=list(ALGOS))
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="default: every run in the YAML")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=2, help="parallel runs (about 1 per CPU core)")
    ap.add_argument("--total-frames", type=int, default=None, help="override, e.g. for a quick test")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true", help="re-run runs that are already done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    extra = []
    if args.total_frames:
        extra += ["--total-frames", str(args.total_frames)]
    if args.set:
        extra += ["--set", *args.set]

    jobs = []
    for algo in args.algos:
        spec = yaml.safe_load((ROOT / "configs" / f"{algo}.yaml").read_text())
        runs = args.runs if args.runs is not None else sorted(int(k) for k in spec["runs"])
        overridden = {"total_frames"} if args.total_frames else set()
        overridden |= {item.partition("=")[0].split(".")[0] for item in args.set}
        for seed in args.seeds:
            for run in runs:
                rid = run_id(run, seed)
                marker = done_marker(algo, rid)
                expected = {k: v for k, v in {**spec["defaults"], **(spec["runs"][run] or {})}.items()
                            if k not in overridden}
                if marker.exists() and not args.force:
                    if is_current(marker, expected):
                        print(f"skip  {algo} {rid} (done)")
                        continue
                    print(f"redo  {algo} {rid} (its settings in configs/{algo}.yaml changed)")
                jobs.append((algo, run, seed, rid))
    print(f"{len(jobs)} runs to do, {args.workers} at a time")
    if args.dry_run:
        for algo, run, seed, rid in jobs:
            print("  ", " ".join(command(algo, run, seed, extra)))
        return

    console = ROOT / "logs" / "console"
    console.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def work(job):
        algo, run, seed, rid = job
        out = console / f"{algo}_{rid}.out"
        start = time.time()
        with out.open("w") as f:
            code = subprocess.call(command(algo, run, seed, extra), cwd=ROOT, stdout=f,
                                   stderr=subprocess.STDOUT)
        final = ""
        for line in out.read_text(errors="replace").splitlines():
            if "final eval" in line:
                final = line.split("final eval", 1)[1].strip()
        return job, code, time.time() - start, final

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            (algo, run, seed, rid), code, secs, final = fut.result()
            status = "ok  " if code == 0 else f"FAIL({code})"
            print(f"[{i}/{len(jobs)}] {status} {algo} {rid} in {secs / 60:.1f} min | {final}"
                  f" | elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
            if code != 0:
                print(f"     see logs/console/{algo}_{rid}.out", flush=True)


if __name__ == "__main__":
    main()
