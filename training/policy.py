"""Load a trained ShambaGuard agent and act with it. No TorchRL needed at run time.

    from training.policy import load_best
    agent = load_best()                       # reads models/best.json
    action = agent.predict(obs, info["action_mask"])

``predict(obs, mask) -> action`` is the whole interface, so the same object can sit behind
a farm dashboard, a FastAPI endpoint or a phone app.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BEST_FILE = ROOT / "models" / "best.json"
ALGOS = ("ppo", "a2c", "reinforce", "dqn")


def model_path(algo: str, run_id: str) -> Path:
    return ROOT / "models" / ("dqn" if algo == "dqn" else "pg") / f"{algo}_{run_id}.pt"


def _mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    """Same layer layout as torchrl.modules.MLP(num_cells=hidden, activation_class=ReLU), so
    its state_dict keys (0.weight, 2.weight, ...) load directly."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _load_checkpoint(path: Path) -> dict:
    # obs_dim / n_actions were saved as numpy ints; allow just those types in the safe loader
    safe = [np._core.multiarray.scalar, np.dtype, *(type(np.dtype(t)) for t in ("int64", "int32"))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu")


class Policy:
    """Greedy policy: argmax of the network's output (policy logits or Q-values) over the
    actions that are valid right now."""

    def __init__(self, algo: str, run_id: str):
        path = model_path(algo, run_id)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (train it, or copy it from Colab)")
        ck = _load_checkpoint(path)
        self.algo, self.run_id = algo, run_id
        self.use_mask = bool(ck.get("action_mask", True))
        self.obs_dim, self.n_actions = int(ck["obs_dim"]), int(ck["n_actions"])
        self.net = _mlp(self.obs_dim, self.n_actions, list(ck["hidden"]))
        self.net.load_state_dict(ck["q"] if algo == "dqn" else ck["policy"])
        self.net.eval()
        cfg = path.with_suffix("").with_suffix(".config.json")
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}

    @property
    def name(self) -> str:
        return f"{self.algo.upper() if self.algo != 'reinforce' else 'REINFORCE'} {self.run_id}"

    @torch.no_grad()
    def scores(self, obs) -> torch.Tensor:
        return self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0))[0]

    def predict(self, obs, action_mask=None) -> int:
        s = self.scores(obs)
        if self.use_mask and action_mask is not None:
            s = s.masked_fill(~torch.as_tensor(np.asarray(action_mask, dtype=bool)), -float("inf"))
        return int(s.argmax().item())


def load_best(path: Path = BEST_FILE) -> Policy:
    best = json.loads(Path(path).read_text())
    return Policy(best["algo"], best["run_id"])
