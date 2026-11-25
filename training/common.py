"""Shared pieces for every trainer: TorchRL env factory, greedy evaluation, CSV logs, configs."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torchrl.data import Binary
from torchrl.envs import GymWrapper, RewardSum, SerialEnv, StepCounter, TransformedEnv
from torchrl.envs.gym_like import default_info_dict_reader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.custom_env import ShambaGuardEnv, make_config  # noqa: E402

EVAL_SEEDS = list(range(1000, 1010))


def eval_seeds(n: int) -> list[int]:
    """Seeds 1000 .. 1000+n-1. Periodic evals use the spec's 10; final evals use more,
    because seasons vary a lot (greedy's std is ~115, so 10 seeds give a SE of ~35)."""
    return list(range(1000, 1000 + int(n)))


# --------------------------------------------------------------------------- env
def env_overrides(cfg: dict) -> dict:
    """The config's optional ``env:`` block, as EnvConfig overrides (YAML lists -> tuples).

    The potential-based shaping discount follows the agent's gamma unless set explicitly,
    which is what keeps the optimal policy unchanged."""
    over = {k: tuple(v) if isinstance(v, list) else v for k, v in (cfg.get("env") or {}).items()}
    if "gamma" in cfg:
        over.setdefault("shaping_gamma", float(cfg["gamma"]))
    return over


def make_env(preset: str, action_mask: bool = True, overrides: dict | None = None):
    def _make():
        env = GymWrapper(ShambaGuardEnv(make_config(preset, **(overrides or {}))),
                         categorical_action_encoding=True)
        if action_mask:  # env puts the valid-action mask in info; expose it as "action_mask"
            n = env.action_spec.space.n
            env.set_info_dict_reader(default_info_dict_reader(
                ["action_mask"], spec={"action_mask": Binary(n, dtype=torch.bool)}))
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(RewardSum())
        return env
    return _make


def make_batched_env(preset: str, num_envs: int, seed: int, action_mask: bool = True,
                     overrides: dict | None = None):
    make = make_env(preset, action_mask, overrides)
    env = SerialEnv(num_envs, make) if num_envs > 1 else make()
    env.set_seed(seed)
    return env


# ------------------------------------------------------------------ evaluation
EVAL_KEYS = ["return", "task_return", "harvest_return", "plots_harvested", "plots_lost", "stranded_rate",
             "water_used", "fertilizer_used", "pesticide_used", "episode_length",
             "total_yield", "invalid_actions"]


@torch.no_grad()
def evaluate(net: nn.Module, preset: str, device, seeds=EVAL_SEEDS,
             action_mask: bool = True, overrides: dict | None = None) -> dict:
    """Greedy policy on fixed seeds: argmax of ``net(obs)`` (policy logits or Q-values)."""
    env = ShambaGuardEnv(make_config(preset, **(overrides or {})))
    rows = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        done = False
        while not done:
            scores = net(torch.as_tensor(obs, device=device).unsqueeze(0))[0]
            if action_mask:
                scores = scores.masked_fill(
                    ~torch.as_tensor(info["action_mask"], device=device), -float("inf"))
            obs, _, term, trunc, info = env.step(int(scores.argmax(-1).item()))
            done = term or trunc
        rows.append([info["episode_return"], info["task_return"], info["harvest_return"], info["plots_harvested"],
                     info["plots_lost"], float(info["stranded"]), info["water_used"],
                     info["fertilizer_used"], info["pesticide_used"], env.t,
                     info["total_yield"], info["invalid_actions"]])
    a = np.array(rows, dtype=float)
    out = {f"eval_{k}": float(a[:, i].mean()) for i, k in enumerate(EVAL_KEYS)}
    out["eval_return_std"] = float(a[:, 0].std())
    out["eval_task_return_std"] = float(a[:, 1].std())
    return out


def eval_line(run_id: str, frames: int, time_s: float, train_return: float, ev: dict,
              extra: str = "") -> str:
    return (f"[{run_id}] {frames:>7d} frames | {time_s:>6.0f}s | train {train_return:7.1f} | "
            f"eval {ev['eval_return']:7.1f} task {ev['eval_task_return']:6.1f} +- "
            f"{ev['eval_task_return_std']:5.1f} | harvested "
            f"{ev['eval_plots_harvested']:4.1f} lost {ev['eval_plots_lost']:4.1f} "
            f"yield {ev['eval_total_yield']:4.1f} | {extra}")


# ------------------------------------------------------------------------ logs
class CsvLog:
    def __init__(self, path: Path):
        self.path, self.fields = path, None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)

    def write(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row)
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


# --------------------------------------------------------------------- configs
def load_runs(path: Path) -> tuple[dict, dict]:
    spec = yaml.safe_load(Path(path).read_text())
    return spec["defaults"], {int(k): (v or {}) for k, v in spec["runs"].items()}


def apply_set(cfg: dict, items: list[str]) -> dict:
    """``--set KEY=VALUE`` overrides; ``env.<field>=value`` goes into the EnvConfig block."""
    cfg = dict(cfg)
    for item in items:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw)
        if key.startswith("env."):
            cfg["env"] = {**(cfg.get("env") or {}), key[4:]: val}
        else:
            cfg[key] = val
    return cfg


def add_run_args(ap) -> None:
    """CLI flags shared by every trainer."""
    ap.add_argument("--sweep", type=Path, default=None, help="run every config in this YAML")
    ap.add_argument("--config", type=Path, default=None, help="YAML (default configs/<algo>.yaml)")
    ap.add_argument("--run", type=int, nargs="*", default=[1], help="run numbers from the YAML")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--total-frames", type=int, default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for run ids, e.g. _seed1")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set gamma=0.995 env.r_growth=0")


def run_configs(args, algo: str):
    """Yield (run_id, cfg) for the runs the CLI selected."""
    yaml_path = args.sweep or args.config or ROOT / "configs" / f"{algo}.yaml"
    defaults, runs = load_runs(yaml_path)
    selected = sorted(runs) if args.sweep else args.run
    print(f"algo={algo} | runs={selected} | config={yaml_path}", flush=True)
    for n in selected:
        cfg = {**defaults, **runs[n]}
        for key in ("total_frames", "preset", "num_envs"):
            val = getattr(args, key)
            if val is not None:
                cfg[key] = val
        yield f"run{n:02d}{args.tag}", apply_set(cfg, args.set)


def print_final(run_id: str, final: dict) -> None:
    print(f"[{run_id}] final eval return {final['eval_return']:.1f} +- "
          f"{final['eval_return_std']:.1f}, task {final['eval_task_return']:.1f} +- "
          f"{final['eval_task_return_std']:.1f}, harvest {final['eval_harvest_return']:.1f}, "
          f"yield {final['eval_total_yield']:.1f}", flush=True)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(name)
    torch.set_num_threads(1 if dev.type == "cpu" else torch.get_num_threads())
    return dev
