"""Demonstrations from the greedy heuristic, and behaviour cloning (BC) for warm starts.

RL from scratch did not learn to tend crops within the 200k-frame budget (see the spec's
"Fixes after the first PPO runs"), so every learner can start from the greedy+fertilizer
heuristic instead:
  * policy-gradient methods load a BC-pretrained policy network (cfg ``init_policy``);
  * DQN pre-fills its replay buffer with demonstration transitions (cfg ``demo_episodes``).

Demonstrations use training seeds from 100_000 up, disjoint from eval seeds 1000-1009.

Examples:
  python training/bc.py                        # 300 demo seasons -> models/bc/bc_greedy.pt
  python training/bc.py --episodes 100 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import MLP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, evaluate, pick_device  # noqa: E402

from environment.baselines import GreedyAgent  # noqa: E402
from environment.custom_env import ShambaGuardEnv, make_config  # noqa: E402

DEMO_SEED0 = 100_000


def collect_demos(preset: str = "standard", overrides: dict | None = None,
                  episodes: int = 300, seed0: int = DEMO_SEED0,
                  use_fertilizer: bool = True) -> dict[str, np.ndarray]:
    """Roll out the greedy heuristic; returns transitions as numpy arrays."""
    cfg = make_config(preset, **(overrides or {}))
    env = ShambaGuardEnv(cfg)
    buf = {k: [] for k in ("obs", "action", "mask", "reward", "next_obs", "next_mask",
                           "terminated", "task_return")}
    for ep in range(episodes):
        agent = GreedyAgent(cfg, use_fertilizer=use_fertilizer)
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            a = agent.act(obs)
            mask = info["action_mask"]
            nobs, r, term, trunc, info = env.step(a)
            buf["obs"].append(obs); buf["action"].append(a); buf["mask"].append(mask)
            buf["reward"].append(r); buf["next_obs"].append(nobs)
            buf["next_mask"].append(info["action_mask"]); buf["terminated"].append(term)
            obs, done = nobs, term or trunc
        buf["task_return"].append(info["task_return"])
    return {k: np.asarray(v) for k, v in buf.items()}


def demos_to_tensordict(d: dict[str, np.ndarray], device) -> TensorDict:
    """Demonstrations in the TorchRL layout the DQN replay buffer stores."""
    n = len(d["action"])
    t = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
    term = t(d["terminated"], torch.bool).unsqueeze(-1)
    return TensorDict({
        "observation": t(d["obs"], torch.float32),
        "action": t(d["action"], torch.int64),
        "action_mask": t(d["mask"], torch.bool),
        "next": TensorDict({
            "observation": t(d["next_obs"], torch.float32),
            "action_mask": t(d["next_mask"], torch.bool),
            "reward": t(d["reward"], torch.float32).unsqueeze(-1),
            "done": term.clone(), "terminated": term.clone(),
            "truncated": torch.zeros_like(term),
        }, batch_size=[n]),
    }, batch_size=[n], device=device)


def train_bc(d: dict[str, np.ndarray], hidden: list[int], device, epochs: int = 20,
             lr: float = 1e-3, batch_size: int = 512, seed: int = 0,
             net: nn.Module | None = None) -> tuple[nn.Module, dict]:
    """Cross-entropy on masked logits (cosine LR decay); pass ``net`` to continue training."""
    torch.manual_seed(seed)
    obs = torch.as_tensor(d["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(d["action"], dtype=torch.int64, device=device)
    mask = torch.as_tensor(d["mask"], dtype=torch.bool, device=device)
    n_actions = mask.shape[-1]
    if net is None:
        net = MLP(in_features=obs.shape[-1], out_features=n_actions, num_cells=hidden,
                  activation_class=nn.ReLU, device=device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.05)
    n = len(act)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = net(obs[idx]).masked_fill(~mask[idx], -1e9)
            loss = nn.functional.cross_entropy(logits, act[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(-1) == act[idx]).sum())
        sched.step()
        stats = dict(epoch=epoch + 1, loss=total / n, accuracy=correct / n)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"[bc] epoch {epoch + 1:3d} | loss {stats['loss']:.4f} | "
                  f"accuracy {stats['accuracy']:.3f}", flush=True)
    return net, stats


@torch.no_grad()
def dagger_states(net: nn.Module, preset: str, episodes: int, seed0: int, device,
                  use_fertilizer: bool = True) -> dict[str, np.ndarray]:
    """DAgger: roll out the current policy, label every visited state with greedy's action."""
    cfg = make_config(preset)
    env = ShambaGuardEnv(cfg)
    obs_l, act_l, mask_l = [], [], []
    for ep in range(episodes):
        expert = GreedyAgent(cfg, use_fertilizer=use_fertilizer)
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            obs_l.append(obs); mask_l.append(info["action_mask"]); act_l.append(expert.act(obs))
            logits = net(torch.as_tensor(obs, device=device)[None])[0]
            logits = logits.masked_fill(~torch.as_tensor(info["action_mask"], device=device), -1e9)
            obs, _, term, trunc, info = env.step(int(logits.argmax()))
            done = term or trunc
    return dict(obs=np.asarray(obs_l), action=np.asarray(act_l), mask=np.asarray(mask_l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="standard")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--no-fertilizer", action="store_true")
    ap.add_argument("--dagger", type=int, default=4, help="DAgger iterations after BC")
    ap.add_argument("--dagger-episodes", type=int, default=100)
    ap.add_argument("--dagger-epochs", type=int, default=15, help="extra epochs per DAgger round")
    ap.add_argument("--final-epochs", type=int, default=80, help="last pass over all data")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "bc" / "bc_greedy.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    t0 = time.time()
    d = collect_demos(args.preset, episodes=args.episodes,
                      use_fertilizer=not args.no_fertilizer)
    print(f"[bc] {len(d['action'])} transitions from {args.episodes} greedy seasons "
          f"(task return {d['task_return'].mean():.1f}) in {time.time() - t0:.0f}s", flush=True)
    net, stats = train_bc(d, args.hidden, device, epochs=args.epochs)
    ev = evaluate(net, args.preset, device)
    print(f"[bc] BC only: task return {ev['eval_task_return']:.1f}", flush=True)
    data = {k: d[k] for k in ("obs", "action", "mask")}
    for it in range(args.dagger):
        new = dagger_states(net, args.preset, args.dagger_episodes,
                            DEMO_SEED0 + 10_000 * (it + 1), device, not args.no_fertilizer)
        data = {k: np.concatenate([data[k], new[k]]) for k in data}
        net, stats = train_bc(data, args.hidden, device, epochs=args.dagger_epochs,
                              seed=it + 1, net=net)
        ev = evaluate(net, args.preset, device)
        print(f"[bc] DAgger {it + 1}/{args.dagger}: {len(data['action'])} states, "
              f"task return {ev['eval_task_return']:.1f}", flush=True)
    if args.final_epochs:
        net, stats = train_bc(data, args.hidden, device, epochs=args.final_epochs,
                              lr=5e-4, seed=99, net=net)
        ev = evaluate(net, args.preset, device)
    np.savez_compressed(args.out.with_name("bc_dataset.npz"), **data)
    print(f"[bc] eval on seeds 1000-1009: task return {ev['eval_task_return']:.1f} +- "
          f"{ev['eval_task_return_std']:.1f}, harvested {ev['eval_plots_harvested']:.1f}, "
          f"lost {ev['eval_plots_lost']:.1f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": net.state_dict(), "obs_dim": d["obs"].shape[-1],
                "n_actions": d["mask"].shape[-1], "hidden": args.hidden, "action_mask": True},
               args.out)
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        vars(args) | {"out": str(args.out)}, bc=stats, eval=ev,
        demo_task_return=float(d["task_return"].mean())), indent=2))
    print(f"[bc] saved {args.out}")


if __name__ == "__main__":
    main()
