"""ShambaGuard: watch the best trained agent tend the farm for one season.

Loads models/best.json (written by scripts/pick_best.py), opens the Pygame window and prints
one line per step in the terminal.

  python main.py                          # best agent, live window + verbose terminal
  python main.py --algo dqn --run 9       # any trained run instead
  python main.py --agent greedy+fert      # a baseline instead of a trained agent
  python main.py --preset drought --seed 7
  python main.py --no-render --episodes 20   # headless: just score some seasons
  python main.py --gif assets/best_agent.gif # save a GIF instead of opening a window

Keys in the window: D toggles the hidden (unscanned) pests, Esc quits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.baselines import GreedyAgent  # noqa: E402
from environment.custom_env import ShambaGuardEnv, make_config, PRESETS  # noqa: E402
from training.policy import BEST_FILE, Policy, load_best  # noqa: E402


def build_agent(args, cfg):
    """Returns (name, act(obs, info) -> action)."""
    if args.agent == "trained":
        pol = Policy(args.algo, f"run{args.run:02d}") if args.algo else load_best()
        score = pol.config.get("final_eval", {}).get("eval_task_return")
        note = f" (final eval task return {score:+.1f} on 50 seeds)" if score is not None else ""
        print(f"Agent: {pol.name}{note}")
        return pol.name, lambda obs, info: pol.predict(obs, info["action_mask"])
    if args.agent == "random":
        rng = np.random.default_rng(args.seed)
        return "random", lambda obs, info: int(rng.choice(np.flatnonzero(info["action_mask"])))
    greedy = GreedyAgent(cfg, use_fertilizer=args.agent == "greedy+fert")
    return args.agent, lambda obs, info: greedy.act(obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["trained", "greedy", "greedy+fert", "random"], default="trained")
    ap.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "dqn"], default=None,
                    help=f"trained run to load instead of {BEST_FILE.relative_to(ROOT)}")
    ap.add_argument("--run", type=int, default=None, help="run number for --algo")
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-render", action="store_true", help="no window")
    ap.add_argument("--gif", default=None, help="save a GIF here instead of opening a window")
    ap.add_argument("--true-state", action="store_true", help="start with hidden pests shown")
    ap.add_argument("--quiet", action="store_true", help="no per-step terminal lines")
    args = ap.parse_args()
    if args.algo and args.run is None:
        ap.error("--algo needs --run")

    cfg = make_config(args.preset)
    mode = None if args.no_render else ("rgb_array" if args.gif else "human")
    env = ShambaGuardEnv(cfg, render_mode=mode,
                         render_kwargs=dict(show_true_state=args.true_state, fps=args.fps,
                                            cell=64 if args.gif else 96))
    name, act = build_agent(args, cfg)

    totals, frames = [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        print(f"\n=== Season {ep + 1}/{args.episodes} | {name} | preset {args.preset} | seed {seed} ===")
        if args.gif:
            frames.append(env.render())
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs, info))
            done = term or trunc
            if not args.quiet:
                print(env.verbose_line())
            if args.gif and (env.t % 2 == 0 or done):
                frames.append(env.render())
            if mode == "human" and env._renderer is not None and env._renderer.window_closed:
                print("window closed")
                env.close()
                return
        why = ("drone stranded (battery empty)" if info["stranded"] else
               "season over" if env.t >= cfg.season_days * cfg.steps_per_day
               else "no crops left (all harvested or lost)")
        print(f"--- {why}: task return {info['task_return']:+.1f} "
              f"(harvest {info['harvest_return']:.1f}) | harvested {info['plots_harvested']} "
              f"lost {info['plots_lost']} | yield {info['total_yield']:.1f} | "
              f"water {info['water_used']} fert {info['fertilizer_used']} pest {info['pesticide_used']}")
        totals.append(info["task_return"])

    if args.episodes > 1:
        t = np.array(totals)
        print(f"\n{name}: mean task return {t.mean():+.1f} ± {t.std():.1f} over {len(t)} seasons")
    if args.gif:
        from PIL import Image
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
        print(f"GIF: {args.gif} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
