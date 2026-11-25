"""Report figures 1-7 (spec "Evaluation, metrics and report figures") -> figures/*.png.

  python scripts/make_figures.py                      # all figures
  python scripts/make_figures.py --only 2 6           # some of them
  python scripts/make_figures.py --best ppo=4 dqn=9   # override the best runs

Needs logs/sweep_summary.csv (scripts/summarize_sweeps.py) and, for Fig 6,
logs/generalization.csv (scripts/generalize.py). Best runs default to the highest final
task return per algorithm; a best run's extra seeds (runNN_seed1, _seed2) are used when
present, and each legend says how many seeds went into the curve.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"
LOGS = ROOT / "logs"

ALGOS = ("ppo", "a2c", "reinforce", "dqn")
NAME = {"ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE", "dqn": "DQN"}
# categorical slots 1-4 of the validated reference palette, fixed per algorithm
COLOR = {"ppo": "#2a78d6", "a2c": "#eb6834", "reinforce": "#1baf7a", "dqn": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BASE_COLOR = {"Greedy + fert": "#52514e", "Greedy": "#8a8984", "Random": "#c3c2b7"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
    "legend.frameon": False, "text.color": INK,
})


# ------------------------------------------------------------------------- data
def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def summary() -> list[dict]:
    with (LOGS / "sweep_summary.csv").open() as f:
        return list(csv.DictReader(f))


def best_runs(overrides: dict[str, int]) -> dict[str, int]:
    best: dict[str, tuple[float, int]] = {}
    for r in summary():
        if "_seed" in r["run_id"]:
            continue
        s = float(r["final_task_return"])
        if r["algo"] not in best or s > best[r["algo"]][0]:
            best[r["algo"]] = (s, int(r["run"]))
    out = {a: run for a, (_, run) in best.items()}
    out.update(overrides)
    return out


def seed_ids(algo: str, run: int) -> list[str]:
    base = f"run{run:02d}"
    return [rid for rid in (base, f"{base}_seed1", f"{base}_seed2")
            if (LOGS / algo / f"{rid}_eval.csv").exists()]


def stacked(algo: str, rids: list[str], col: str, kind: str = "eval"):
    """(frames, values[n_seeds, n_points]) on the shortest common frame grid."""
    curves = []
    for rid in rids:
        d = read_csv(LOGS / algo / (f"{rid}_eval.csv" if kind == "eval" else f"{rid}.csv"))
        curves.append((d["frames"], d[col]))
    n = min(len(f) for f, _ in curves)
    return curves[0][0][:n], np.stack([v[:n] for _, v in curves])


def baseline_returns() -> dict[str, float]:
    """Greedy / random on the final-eval seeds (from the generalization run, else spec values)."""
    out = {"Greedy + fert": 162.8, "Greedy": 106.8, "Random": -32.0}
    path = LOGS / "generalization.csv"
    if path.exists():
        for r in csv.DictReader(path.open()):
            if r["condition"] == "train seeds" and r["agent"] in out:
                out[r["agent"]] = float(r["task_return"])
    return out


def ref_line(ax, y, label, color, x_text=1.0, va="bottom"):
    ax.axhline(y, color=color, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (x_text, y), xycoords=("axes fraction", "data"), ha="right", va=va,
                fontsize=8.5, color=INK2, xytext=(0, 2 if va == "bottom" else -2),
                textcoords="offset points",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))


def legend_below(ax, ncol=2):
    """Legend under the plot area, so it never covers the lines."""
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=ncol, fontsize=8.5,
              handlelength=2.2, columnspacing=1.4)


def kframes(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("Environment frames")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", (FIG / name).relative_to(ROOT))


# --------------------------------------------------------------------- figure 1
def fig1():
    """Mid-season screenshot of the greedy+fert agent, with numbered callouts."""
    from environment.baselines import GreedyAgent
    from environment.custom_env import ShambaGuardEnv, make_config

    cfg = make_config("standard")
    cell = 96
    env = ShambaGuardEnv(cfg, render_mode="rgb_array",
                         render_kwargs=dict(show_true_state=False, cell=cell))
    agent = GreedyAgent(cfg, use_fertilizer=True)
    # pick the frame (seed 3, days 20-34) that shows the most kinds of plot state at once
    best_score, best = -1, None
    obs, info = env.reset(seed=3)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        if 18 <= env.day <= 34 and env.step_in_day % 2 == 0:
            kinds = _plot_kinds(env)
            score = sum(1 for v in kinds.values() if v) + (env.last_action in (1, 2, 3, 4, 5))
            if score > best_score:
                best_score, best = score, (env.render(), kinds, env.pos)
    frame, kinds, pos = best
    env.close()

    margin = max(cell // 6, 10)
    centre = lambda rc, dy=0.44: (margin + rc[1] * cell + cell / 2, margin + rc[0] * cell + cell * dy)  # noqa: E731
    callouts = [("Base: recharge + refill water", (0, 0), 0.5),
                ("Drone (battery LED at centre)", tuple(pos), 0.42)]
    for key, label in (("mature", "Mature crop (gold ring): harvest"),
                       ("pest", "Known pests (red specks)"),
                       ("dry", "Dry soil (light) / low moisture gauge"),
                       ("harvested", "Harvested plot (stubble)"),
                       ("dead", "Dead crop (X)"),
                       ("young", "Young crop (small; colour = health)")):
        if kinds[key]:
            rc = min(kinds[key], key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
            callouts.append((label, rc, 0.44))
    grid_r = margin + 6 * cell
    hud_x = 2 * margin + 6 * cell

    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100 + 3.2, h / 100))
    ax.imshow(frame)
    ax.set_axis_off()
    used = set()
    for i, (label, rc, dy) in enumerate(callouts, 1):
        if rc in used and i > 2:
            continue
        used.add(rc)
        x, y = centre(rc, dy)
        off = 0.28 * cell if i == 2 else -0.3 * cell
        ax.annotate(str(i), (x + off, y - 0.3 * cell), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="#2a78d6", ec="white", lw=1.2))
    hud = [("Day, step and weather", 0.11), ("Battery / water / fertilizer / pesticide", 0.24),
           ("Last action, step reward, return", 0.43), ("Legend", 0.6)]
    n0 = len(callouts)
    for j, (label, fy) in enumerate(hud, 1):
        ax.annotate(str(n0 + j), (hud_x - 8, fy * h), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="#eb6834", ec="white", lw=1.2))
    lines = [f"{i}  {label}" for i, (label, _, _) in enumerate(callouts, 1)]
    lines += [f"{n0 + j}  HUD: {label}" for j, (label, _) in enumerate(hud, 1)]
    lines += ["", "Bars under each plot: moisture (blue)", "and nutrient (green); white ticks",
              "mark the ideal band."]
    ax.text(w + 20, 10, "\n".join(lines), va="top", ha="left", fontsize=9, color=INK,
            linespacing=1.6)
    ax.set_xlim(0, w + 330)
    ax.set_title("Fig 1. ShambaGuard: 6x6 farm (left) and HUD (right), greedy+fert agent, "
                 "standard preset, seed 3", loc="left")
    _ = grid_r
    save(fig, "fig1_environment.png")


def _plot_kinds(env) -> dict[str, list]:
    n = env.n
    known = env.known_pest()
    out = {k: [] for k in ("mature", "pest", "dry", "harvested", "dead", "young")}
    lo = env.cfg.moisture_band[0]
    for r in range(n):
        for c in range(n):
            if (r, c) == (0, 0) or (r, c) == tuple(env.pos):
                continue
            if env.crop[r, c]:
                if env.growth[r, c] >= 1.0:
                    out["mature"].append((r, c))
                elif known[r, c] > 0:
                    out["pest"].append((r, c))
                elif env.moisture[r, c] < lo:
                    out["dry"].append((r, c))
                elif env.growth[r, c] < 0.5:
                    out["young"].append((r, c))
            elif env.status[r, c] == 1:
                out["harvested"].append((r, c))
            elif env.status[r, c] == 2:
                out["dead"].append((r, c))
    return out


# --------------------------------------------------------------------- figure 2
def fig2(best):
    base = baseline_returns()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for ax, title, which in ((axes[0], "(a) Best run per algorithm (warm start)", "best"),
                             (axes[1], "(b) Run 0: trained from scratch", "scratch")):
        for algo in ALGOS:
            run = best.get(algo) if which == "best" else 0
            rids = seed_ids(algo, run) if run is not None else []
            if not rids:
                continue
            x, y = stacked(algo, rids, "eval_task_return")
            m, s = y.mean(0), y.std(0)
            label = f"{NAME[algo]} run {run} ({len(rids)} seed{'s' if len(rids) > 1 else ''})"
            ax.plot(x, m, color=COLOR[algo], label=label, zorder=3)
            if len(rids) > 1:
                ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.15, lw=0, zorder=2)
        ref_line(ax, base["Greedy + fert"], f"greedy + fert {base['Greedy + fert']:+.0f}", INK2)
        ref_line(ax, base["Random"], f"random {base['Random']:+.0f}", MUTED, va="top")
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, base["Random"] - 20), hi)  # room for the random label
        ax.set_title(title, loc="left")
        ax.set_ylabel("Task return per season (eval, 10 seeds)")
        kframes(ax)
        legend_below(ax)
    fig.suptitle("Fig 2. Cumulative task reward per episode during training "
                 "(mean ± std over training seeds; shaping excluded)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_reward_curves.png")


# --------------------------------------------------------------------- figure 3
def fig3(best):
    run = best.get("dqn")
    if run is None:
        print("fig3: no DQN runs yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for r, style, lab in ((run, "-", f"run {run} (best, warm start)"), (0, (0, (4, 2)), "run 0 (from scratch)")):
        rids = seed_ids("dqn", r)
        if not rids:
            continue
        for ax, col in zip(axes, ("td_loss", "mean_q")):
            x, y = stacked("dqn", rids, col, kind="train")
            m = y.mean(0)
            lw = 2 if style == "-" else 1.6
            ax.plot(x, m, color=COLOR["dqn"] if style == "-" else INK2, ls=style, lw=lw,
                    label=f"{lab}, {len(rids)} seed{'s' if len(rids) > 1 else ''}")
            if len(rids) > 1:
                ax.fill_between(x, m - y.std(0), m + y.std(0), color=COLOR["dqn"], alpha=0.15, lw=0)
    axes[0].set_title("(a) TD loss (Huber, per update)", loc="left")
    axes[0].set_ylabel("TD loss")
    axes[1].set_title("(b) Mean Q-value of the batch", loc="left")
    axes[1].set_ylabel("Mean Q")
    for ax in axes:
        kframes(ax)
        legend_below(ax)
    fig.suptitle("Fig 3. DQN training diagnostics", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_dqn_loss_q.png")


# --------------------------------------------------------------------- figure 4
def fig4(best):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, which, title in ((axes[0], "best", "(a) Policy entropy, best runs (warm start)"),
                             (axes[1], "scratch", "(b) Policy entropy, run 0 (from scratch)")):
        for algo in ("ppo", "a2c", "reinforce"):
            run = best.get(algo) if which == "best" else 0
            rids = seed_ids(algo, run) if run is not None else []
            if not rids:
                continue
            x, y = stacked(algo, rids, "entropy", kind="train")
            m = np.nanmean(y, 0)
            ax.plot(x, m, color=COLOR[algo], label=f"{NAME[algo]} run {run}")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Entropy (nats, over valid actions)")
        kframes(ax)
        legend_below(ax, ncol=3)
    ax = axes[2]
    for r, style, lab in ((best.get("dqn"), "-", "best run"), (0, (0, (4, 2)), "from scratch")):
        if r is None or not seed_ids("dqn", r):
            continue
        x, y = stacked("dqn", seed_ids("dqn", r)[:1], "epsilon", kind="train")
        ax.plot(x, y[0], color=COLOR["dqn"] if style == "-" else INK2, ls=style,
                label=f"DQN run {r} ({lab})")
    ax.set_title("(c) DQN exploration: ε schedule", loc="left")
    ax.set_ylabel("ε (probability of a random action)")
    kframes(ax)
    legend_below(ax)
    fig.suptitle("Fig 4. Exploration: policy entropy (policy-gradient methods) vs ε (DQN)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_entropy.png")


# --------------------------------------------------------------------- figure 5
def frames_to_80(x, y) -> float:
    """Frame from which the 3-point smoothed eval curve stays at or above 80% of its final
    level for the rest of training (final = mean of the last 3 evals). "Reaches and stays"
    rather than "first reaches": a warm-started agent is above 80% at frame 0, so the
    first-touch version is always ~0 and says nothing. NaN when the final level is not positive."""
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    ys[0], ys[-1] = y[0], y[-1]
    final = float(np.mean(y[-3:]))
    if final <= 0:
        return np.nan
    below = np.flatnonzero(ys < 0.8 * final)
    if not len(below):
        return float(x[0])
    return float(x[below[-1] + 1]) if below[-1] + 1 < len(x) else np.nan


def fig5(best):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    rows = []
    for algo in ALGOS:
        for which, run in (("best", best.get(algo)), ("scratch", 0)):
            rids = seed_ids(algo, run) if run is not None else []
            vals, finals = [], []
            for rid in rids:
                d = read_csv(LOGS / algo / f"{rid}_eval.csv")
                vals.append(frames_to_80(d["frames"], d["eval_task_return"]))
                finals.append(float(np.mean(d["eval_task_return"][-3:])))
            rows.append((algo, which, run, vals, finals))
    ax = axes[0]
    labels, ypos = [], []
    for i, (algo, which, run, vals, finals) in enumerate([r for r in rows if r[1] == "best"]):
        v = np.array(vals, dtype=float)
        mean = np.nanmean(v) if np.isfinite(v).any() else np.nan
        y = len(ALGOS) - 1 - i
        if np.isfinite(mean):
            ax.barh(y, mean, color=COLOR[algo], height=0.55)
            if np.isfinite(v).sum() > 1:
                ax.errorbar(mean, y, xerr=np.nanstd(v), color=INK, capsize=3, lw=1)
            ax.annotate(f"{mean / 1000:.0f}k frames  (final {np.mean(finals):+.0f})", (mean + (np.nanstd(v) if np.isfinite(v).sum() > 1 else 0), y),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=INK2)
        labels.append(f"{NAME[algo]} run {run} ({len(vals)} seed{'s' if len(vals) > 1 else ''})")
        ypos.append(y)
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 230_000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("Frames until eval return stays at or above 80% of its final level")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Warm-started best runs", loc="left")

    ax = axes[1]
    for i, (algo, which, run, vals, finals) in enumerate([r for r in rows if r[1] == "scratch"]):
        y = len(ALGOS) - 1 - i
        f = np.mean(finals) if finals else np.nan
        ax.annotate(f"{NAME[algo]}: never converged (final eval {f:+.0f}, below 0)"
                    if np.isnan(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
                    else f"{NAME[algo]}: {np.nanmean(vals) / 1000:.0f}k frames",
                    (0.02, (y + 0.5) / len(ALGOS)), xycoords="axes fraction", va="center",
                    fontsize=9.5, color=INK)
    ax.set_axis_off()
    ax.set_title("(b) Run 0, from scratch", loc="left")
    fig.suptitle("Fig 5. Convergence speed. Warm-started agents begin at the clone's level, so "
                 "this measures when their eval return settles (stops dipping below 80% of final)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig5_convergence.png")


# --------------------------------------------------------------------- figure 6
def fig6():
    path = LOGS / "generalization.csv"
    if not path.exists():
        print("fig6: run scripts/generalize.py first")
        return
    rows = list(csv.DictReader(path.open()))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    # average the seeds of each algorithm's best run; SE from per-episode std
    agg = {}
    for c in conds:
        for a in agents:
            rs = [r for r in rows if r["condition"] == c and r["agent"] == a]
            m = np.array([float(r["task_return"]) for r in rs])
            sd = np.array([float(r["task_return_std"]) for r in rs])
            n = sum(int(r["episodes"]) for r in rs)
            agg[c, a] = (m.mean(), np.sqrt(np.mean(sd ** 2) / n), len(rs), rs[0]["model"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    k = len(agents)
    width = 0.8 / k
    for j, a in enumerate(agents):
        algo = {v: kk for kk, v in NAME.items()}.get(a)
        color = COLOR[algo] if algo else BASE_COLOR.get(a, MUTED)
        xs = np.arange(len(conds)) + (j - (k - 1) / 2) * width
        ms = [agg[c, a][0] for c in conds]
        es = [1.96 * agg[c, a][1] for c in conds]
        n_models = agg[conds[0], a][2]
        label = a if not algo else f"{a} run {int(agg[conds[0], a][3][3:5])} ({n_models} seed{'s' if n_models > 1 else ''})"
        ax.bar(xs, ms, width * 0.9, color=color, label=label, yerr=es, capsize=2,
               error_kw=dict(lw=0.8, ecolor=INK2))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(conds)),
                  [{"train seeds": "Final-eval seeds\n(1000-1049)", "unseen seeds": "Unseen seeds\n(5000-5049)",
                    "drought": "Drought\n(rarer rain, faster drying)", "high pest": "High pest\n(outbreaks x3)"}.get(c, c)
                   for c in conds])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Task return per season (50 episodes)")
    ax.legend(ncol=4, fontsize=8.5, loc="upper right")
    ax.set_title("Fig 6. Generalization: greedy policies on seeds and conditions never trained on "
                 "(error bars: 95% CI)", loc="left")
    lo = min(agg[c, a][0] - 1.96 * agg[c, a][1] for c in conds for a in agents)
    hi = max(agg[c, a][0] + 1.96 * agg[c, a][1] for c in conds for a in agents)
    ax.set_ylim(min(lo * 1.1, -10), hi * 1.25)
    fig.tight_layout()
    save(fig, "fig6_generalization.png")


# --------------------------------------------------------------------- figure 7
def fig7():
    rows = [r for r in summary() if "_seed" not in r["run_id"]]
    base = baseline_returns()
    present = [a for a in ALGOS if any(r["algo"] == a for r in rows)]
    nmax = max(sum(1 for r in rows if r["algo"] == a) for a in present)
    fig, axes = plt.subplots(2, 2, figsize=(14, 0.32 * nmax * 2 + 2.2), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS):
        rs = sorted((r for r in rows if r["algo"] == algo), key=lambda r: int(r["run"]))
        if not rs:
            ax.set_axis_off()
            continue
        y = np.arange(len(rs))[::-1]
        vals = np.array([float(r["final_task_return"]) for r in rs])
        se = np.array([float(r["final_std"]) for r in rs]) / np.sqrt(50)
        best_i = int(np.argmax(vals))
        colors = [MUTED if r["run"] == "0" else COLOR[algo] for r in rs]
        ax.barh(y, vals, xerr=1.96 * se, color=colors, height=0.68,
                error_kw=dict(lw=0.8, ecolor=INK2, capsize=2))
        for yi, v, r, i in zip(y, vals, rs, range(len(rs))):
            # label just past the end of the bar or its CI whisker, whichever is further right
            ax.annotate(f"{v:+.0f}" + ("  best" if i == best_i else ""),
                        (max(v + 1.96 * se[i], 0), yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK if i == best_i else INK2,
                        fontweight="bold" if i == best_i else "normal")
        labels = []
        for r in rs:
            ch = r["changes"].replace("init_policy=None, ", "").replace("critic_warmup_batches", "warmup")
            ch = "from scratch" if r["run"] == "0" else ch
            ch = ch.replace("frames_per_batch", "fpb").replace("minibatch", "mb")
            labels.append(f"{int(r['run'])}: {ch if len(ch) <= 48 else ch[:47] + '…'}")
        ax.set_yticks(y, labels, fontsize=8)
        ax.axvline(base["Greedy + fert"], color=INK2, lw=1.1, ls=(0, (4, 3)))
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(NAME[algo], loc="left", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Final task return (50 eval seeds, 95% CI)")
    axes[0, 1].annotate(f"greedy + fert {base['Greedy + fert']:+.0f}", (base["Greedy + fert"], 1.0),
                        xycoords=("data", "axes fraction"), ha="right", va="bottom", fontsize=8,
                        color=INK2, xytext=(-3, 2), textcoords="offset points")
    fig.suptitle("Fig 7. Hyperparameter sweeps: final return per run (run 0 grey = from scratch; "
                 "runs 2-9 change one setting from run 1)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_sweeps.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--best", nargs="*", default=[], metavar="ALGO=RUN")
    args = ap.parse_args()
    best = best_runs({k: int(v) for k, v in (s.split("=") for s in args.best)})
    print("best runs:", best)
    figs = {1: fig1, 2: lambda: fig2(best), 3: lambda: fig3(best), 4: lambda: fig4(best),
            5: lambda: fig5(best), 6: fig6, 7: fig7}
    for i in args.only:
        figs[i]()


if __name__ == "__main__":
    main()
