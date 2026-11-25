"""Policy-gradient training: PPO and A2C with TorchRL, REINFORCE as a Monte-Carlo loop.

Examples:
  python training/pg_training.py --algo ppo --run 1                    # one run from configs/ppo.yaml
  python training/pg_training.py --algo ppo --sweep configs/ppo.yaml   # all runs
  python training/pg_training.py --algo a2c --run 1                    # configs/a2c.yaml
  python training/pg_training.py --algo reinforce --run 1              # configs/reinforce.yaml
  python training/pg_training.py --algo ppo --run 1 --set init_policy=models/bc/bc_greedy.pt
  python training/pg_training.py --algo ppo --run 1 --total-frames 30000 --preset easy   # quick check
  python training/pg_training.py --algo ppo --run 1 --set action_mask=false env.r_growth=0

Outputs:
  logs/<algo>/<run_id>.csv        training metrics per collected batch
  logs/<algo>/<run_id>_eval.csv   greedy evaluation on seeds 1000-1009 every eval_every frames
  models/pg/<algo>_<run_id>.pt    final weights (+ <algo>_<run_id>.config.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
try:  # torchrl >= 0.12 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, CsvLog, add_run_args, env_overrides, eval_line, eval_seeds,  # noqa: E402
                    evaluate, make_batched_env, pick_device, print_final, run_configs)

from environment.custom_env import ShambaGuardEnv, make_config  # noqa: E402


# ---------------------------------------------------------------------- models
def build_models(obs_dim: int, n_actions: int, hidden: list[int], action_spec, device,
                 action_mask: bool = True):
    policy_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=hidden,
                     activation_class=nn.ReLU, device=device)
    value_net = MLP(in_features=obs_dim, out_features=1, num_cells=hidden,
                    activation_class=nn.ReLU, device=device)
    if action_mask:  # invalid actions get probability 0 (in collection and in the loss)
        dist_keys, dist_class = {"logits": "logits", "mask": "action_mask"}, MaskedCategorical
    else:
        dist_keys, dist_class = ["logits"], Categorical
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=dist_keys, spec=action_spec, distribution_class=dist_class,
        return_log_prob=True)
    critic = ValueOperator(value_net, in_keys=["observation"])
    return policy_net, value_net, actor, critic


# --------------------------------------------------------------------- helpers
def warm_start(policy_net: nn.Module, cfg: dict, run_id: str, device) -> None:
    """Load a pretrained policy (cfg ``init_policy``, e.g. models/bc/bc_greedy.pt)."""
    if cfg.get("init_policy"):
        ck = torch.load(ROOT / cfg["init_policy"], map_location=device)
        policy_net.load_state_dict(ck["policy"])
        print(f"[{run_id}] policy initialised from {cfg['init_policy']}", flush=True)


def save_run(algo: str, run_id: str, cfg: dict, seed: int, final: dict, best_eval: float,
             t0: float, policy_net: nn.Module, value_net: nn.Module | None,
             obs_dim: int, n_actions: int, use_mask: bool) -> None:
    model_dir = ROOT / "models" / "pg"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy_net.state_dict(),
                "value": value_net.state_dict() if value_net is not None else None,
                "obs_dim": obs_dim, "n_actions": n_actions, "hidden": cfg["hidden"],
                "action_mask": use_mask}, model_dir / f"{algo}_{run_id}.pt")
    (model_dir / f"{algo}_{run_id}.config.json").write_text(json.dumps(
        dict(cfg, algo=algo, run_id=run_id, seed=seed, final_eval=final, best_eval=best_eval,
             train_time_s=round(time.time() - t0, 1)), indent=2))


def final_eval(net: nn.Module, cfg: dict, device, use_mask: bool, env_over: dict) -> dict:
    return evaluate(net, cfg["preset"], device, action_mask=use_mask, overrides=env_over,
                    seeds=eval_seeds(cfg.get("final_eval_episodes", 50)))


# ------------------------------------------------------------------------- A2C
def train_a2c(cfg: dict, run_id: str, seed: int, device: torch.device) -> dict:
    """Synchronous advantage actor-critic: one gradient step per collected batch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_mask = bool(cfg.get("action_mask", True))
    env_over = env_overrides(cfg)
    env = make_batched_env(cfg["preset"], cfg["num_envs"], seed, use_mask, env_over)
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.space.n
    policy_net, value_net, actor, critic = build_models(
        obs_dim, n_actions, cfg["hidden"], env.action_spec, device, use_mask)
    warm_start(policy_net, cfg, run_id, device)
    warmup = int(cfg.get("critic_warmup_batches", 0))

    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"], device=device)
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic,
                    average_gae=bool(cfg.get("normalize_advantage", False)))
    loss_fn = A2CLoss(actor_network=actor, critic_network=critic,
                      entropy_bonus=cfg["entropy_coef"] > 0, entropy_coeff=cfg["entropy_coef"],
                      critic_coeff=cfg["value_coef"], loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    log_dir = ROOT / "logs" / "a2c"
    train_log, eval_log = CsvLog(log_dir / f"{run_id}.csv"), CsvLog(log_dir / f"{run_id}_eval.csv")
    frames, next_eval, t0, best_eval = 0, 0, time.time(), -float("inf")
    recent: list[float] = []
    stats_sum: dict[str, float] = {}
    n_stats = 0
    for batch_idx, batch in enumerate(collector):
        frames += batch.numel()
        with torch.no_grad():
            advantage(batch)
        losses = loss_fn(batch.reshape(-1))
        if batch_idx < warmup:
            loss = losses["loss_critic"]
        else:
            loss = losses["loss_objective"] + losses["loss_critic"]
            if "loss_entropy" in losses.keys():
                loss = loss + losses["loss_entropy"]
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
        optim.step()
        for k, v in losses.items():
            if v.numel() == 1:
                stats_sum[k] = stats_sum.get(k, 0.0) + float(v.detach())
        n_stats += 1
        done = batch["next", "done"].squeeze(-1)
        recent += batch["next", "episode_reward"].squeeze(-1)[done].tolist()

        if frames >= next_eval or frames >= cfg["total_frames"]:
            stats = {k: round(v / max(n_stats, 1), 5) for k, v in stats_sum.items()}
            row = dict(frames=frames, time_s=round(time.time() - t0, 1), train_episodes=len(recent),
                       train_return=float(np.mean(recent)) if recent else float("nan"), **stats)
            train_log.write(row)
            ev = evaluate(policy_net, cfg["preset"], device, action_mask=use_mask, overrides=env_over)
            eval_log.write(dict(frames=frames, time_s=row["time_s"], **ev))
            best_eval = max(best_eval, ev["eval_task_return"])
            print(eval_line(run_id, frames, row["time_s"], row["train_return"], ev,
                            f"entropy {stats.get('entropy', float('nan')):.3f}"), flush=True)
            recent, stats_sum, n_stats = [], {}, 0
            next_eval += cfg["eval_every"]

    collector.shutdown()
    final = final_eval(policy_net, cfg, device, use_mask, env_over)
    save_run("a2c", run_id, cfg, seed, final, best_eval, t0, policy_net, value_net,
             obs_dim, n_actions, use_mask)
    return final


# ------------------------------------------------------------------- REINFORCE
def train_reinforce(cfg: dict, run_id: str, seed: int, device: torch.device) -> dict:
    """Monte-Carlo policy gradient over whole episodes (plain Gymnasium loop).

    baseline: "none" (A = G) or "value" (A = G - V(s), V fitted to G). With
    return_normalization, advantages are standardised within each update."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_mask = bool(cfg.get("action_mask", True))
    env_over = env_overrides(cfg)
    env = ShambaGuardEnv(make_config(cfg["preset"], **env_over))
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    policy_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=cfg["hidden"],
                     activation_class=nn.ReLU, device=device)
    use_baseline = cfg.get("baseline", "none") == "value"
    value_net = (MLP(in_features=obs_dim, out_features=1, num_cells=cfg["hidden"],
                     activation_class=nn.ReLU, device=device) if use_baseline else None)
    warm_start(policy_net, cfg, run_id, device)
    params = list(policy_net.parameters()) + (list(value_net.parameters()) if use_baseline else [])
    optim = torch.optim.Adam(params, lr=cfg["lr"])
    gamma, k_eps = cfg["gamma"], int(cfg["episodes_per_update"])

    log_dir = ROOT / "logs" / "reinforce"
    train_log, eval_log = CsvLog(log_dir / f"{run_id}.csv"), CsvLog(log_dir / f"{run_id}_eval.csv")
    frames, next_eval, t0, best_eval, ep_count = 0, 0, time.time(), -float("inf"), 0
    recent: list[float] = []
    obs, info = env.reset(seed=seed)
    while frames < cfg["total_frames"]:
        obs_b, act_b, mask_b, ret_b = [], [], [], []
        for _ in range(k_eps):
            o_l, a_l, m_l, r_l = [], [], [], []
            done = False
            while not done:
                with torch.no_grad():
                    logits = policy_net(torch.as_tensor(obs, device=device)[None])[0]
                    m = torch.as_tensor(info["action_mask"], device=device)
                    if use_mask:
                        logits = logits.masked_fill(~m, -1e9)
                    a = int(Categorical(logits=logits).sample())
                o_l.append(obs); a_l.append(a); m_l.append(info["action_mask"])
                obs, r, term, trunc, info = env.step(a)
                r_l.append(r)
                done = term or trunc
            recent.append(info["episode_return"])
            g, rets = 0.0, np.zeros(len(r_l))
            for t in range(len(r_l) - 1, -1, -1):   # discounted return-to-go
                g = r_l[t] + gamma * g
                rets[t] = g
            obs_b += o_l; act_b += a_l; mask_b += m_l; ret_b.append(rets)
            frames += len(r_l)
            ep_count += 1
            obs, info = env.reset()
        o = torch.as_tensor(np.asarray(obs_b), dtype=torch.float32, device=device)
        a = torch.as_tensor(act_b, dtype=torch.int64, device=device)
        msk = torch.as_tensor(np.asarray(mask_b), dtype=torch.bool, device=device)
        g = torch.as_tensor(np.concatenate(ret_b), dtype=torch.float32, device=device)
        logits = policy_net(o)
        if use_mask:
            logits = logits.masked_fill(~msk, -1e9)
        dist = Categorical(logits=logits)
        logp, entropy = dist.log_prob(a), dist.entropy().mean()
        loss_v = torch.zeros((), device=device)
        if use_baseline:
            v = value_net(o).squeeze(-1)
            adv = g - v.detach()
            loss_v = nn.functional.smooth_l1_loss(v, g)
        else:
            adv = g
        if cfg.get("return_normalization", True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_pi = -(logp * adv).mean() - cfg["entropy_coef"] * entropy
        loss = loss_pi + cfg.get("value_coef", 0.5) * loss_v
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        optim.step()

        if frames >= next_eval or frames >= cfg["total_frames"]:
            row = dict(frames=frames, time_s=round(time.time() - t0, 1), episodes=ep_count,
                       train_return=float(np.mean(recent)) if recent else float("nan"),
                       loss_policy=round(float(loss_pi.detach()), 5),
                       loss_value=round(float(loss_v.detach()), 5),
                       entropy=round(float(entropy.detach()), 5))
            train_log.write(row)
            ev = evaluate(policy_net, cfg["preset"], device, action_mask=use_mask, overrides=env_over)
            eval_log.write(dict(frames=frames, time_s=row["time_s"], **ev))
            best_eval = max(best_eval, ev["eval_task_return"])
            print(eval_line(run_id, frames, row["time_s"], row["train_return"], ev,
                            f"entropy {row['entropy']:.3f}"), flush=True)
            recent = []
            next_eval += cfg["eval_every"]

    final = final_eval(policy_net, cfg, device, use_mask, env_over)
    save_run("reinforce", run_id, cfg, seed, final, best_eval, t0, policy_net, value_net,
             obs_dim, n_actions, use_mask)
    return final


# ------------------------------------------------------------------------- PPO
def train_ppo(cfg: dict, run_id: str, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_mask = bool(cfg.get("action_mask", True))
    env_over = env_overrides(cfg)
    env = make_batched_env(cfg["preset"], cfg["num_envs"], seed, use_mask, env_over)
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.space.n
    policy_net, value_net, actor, critic = build_models(
        obs_dim, n_actions, cfg["hidden"], env.action_spec, device, use_mask)
    warm_start(policy_net, cfg, run_id, device)
    # Critic-only updates first, so a random critic's advantages don't wreck a warm-started policy.
    warmup = int(cfg.get("critic_warmup_batches", 0))

    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"], device=device)
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg["frames_per_batch"], device=device),
        sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch"])
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic,
                    average_gae=True)
    loss_fn = ClipPPOLoss(actor_network=actor, critic_network=critic,
                          clip_epsilon=cfg["clip_epsilon"], entropy_bonus=cfg["entropy_coef"] > 0,
                          entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["value_coef"],
                          loss_critic_type="smooth_l1",
                          normalize_advantage=bool(cfg.get("normalize_advantage", False)))
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    log_dir = ROOT / "logs" / "ppo"
    train_log = CsvLog(log_dir / f"{run_id}.csv")
    eval_log = CsvLog(log_dir / f"{run_id}_eval.csv")

    frames, next_eval, t0 = 0, 0, time.time()
    best_eval = -float("inf")
    for batch_idx, batch in enumerate(collector):
        frames += batch.numel()
        with torch.no_grad():
            advantage(batch)
        flat = batch.reshape(-1)
        buffer.extend(flat)
        stats = {}
        n_updates = 0
        for _ in range(cfg["epochs"]):
            for _ in range(max(flat.numel() // cfg["minibatch"], 1)):
                mb = buffer.sample()
                losses = loss_fn(mb)
                if batch_idx < warmup:
                    loss = losses["loss_critic"]
                else:
                    loss = losses["loss_objective"] + losses["loss_critic"]
                    if "loss_entropy" in losses.keys():
                        loss = loss + losses["loss_entropy"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                optim.step()
                n_updates += 1
                for k, v in losses.items():
                    if v.numel() == 1:
                        stats[k] = stats.get(k, 0.0) + float(v.detach())
        stats = {k: v / n_updates for k, v in stats.items()}

        done = batch["next", "done"].squeeze(-1)
        ep_returns = batch["next", "episode_reward"].squeeze(-1)[done]
        row = dict(frames=frames, time_s=round(time.time() - t0, 1),
                   train_episodes=int(done.sum()),
                   train_return=float(ep_returns.mean()) if done.any() else float("nan"),
                   **{k: round(v, 5) for k, v in stats.items()})
        train_log.write(row)

        if frames >= next_eval or frames >= cfg["total_frames"]:
            ev = evaluate(policy_net, cfg["preset"], device, action_mask=use_mask, overrides=env_over)
            eval_log.write(dict(frames=frames, time_s=row["time_s"], **ev))
            best_eval = max(best_eval, ev["eval_task_return"])
            print(eval_line(run_id, frames, row["time_s"], row["train_return"], ev,
                            f"entropy {stats.get('entropy', float('nan')):.3f}"), flush=True)
            next_eval += cfg["eval_every"]

    collector.shutdown()
    final = final_eval(policy_net, cfg, device, use_mask, env_over)
    save_run("ppo", run_id, cfg, seed, final, best_eval, t0, policy_net, value_net,
             obs_dim, n_actions, use_mask)
    return final


# ------------------------------------------------------------------------- CLI
TRAINERS = {"ppo": train_ppo, "a2c": train_a2c, "reinforce": train_reinforce}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=sorted(TRAINERS), default="ppo")
    add_run_args(ap)
    args = ap.parse_args()
    device = pick_device(args.device)
    for run_id, cfg in run_configs(args, args.algo):
        print_final(run_id, TRAINERS[args.algo](cfg, run_id, args.seed, device))


if __name__ == "__main__":
    main()
