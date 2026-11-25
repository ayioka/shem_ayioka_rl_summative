"""Watch a baseline agent play one season, or save it as a GIF / screenshots.

Examples:
  python scripts/visualize.py                                  # live window, greedy agent
  python scripts/visualize.py --agent random --fps 30
  python scripts/visualize.py --gif assets/greedy_season.gif   # headless GIF
  python scripts/visualize.py --snapshots 0 12 24 36 --gif ""  # PNGs at chosen days
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.baselines import GreedyAgent, RandomAgent  # noqa: E402
from environment.custom_env import ShambaGuardEnv, make_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["greedy", "greedy+fert", "random"], default="greedy+fert")
    ap.add_argument("--preset", default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--true-state", action="store_true", help="also draw hidden pests")
    ap.add_argument("--gif", default=None, help="save a GIF here instead of opening a window")
    ap.add_argument("--every", type=int, default=2, help="GIF: keep every Nth step")
    ap.add_argument("--cell", type=int, default=None, help="cell size in pixels")
    ap.add_argument("--snapshots", type=int, nargs="*", default=[],
                    help="save PNGs at these days (headless)")
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--verbose", action="store_true", help="print one line per step")
    args = ap.parse_args()

    cfg = make_config(args.preset)
    headless = args.gif is not None or bool(args.snapshots)
    cell = args.cell or (64 if args.gif else 96)
    env = ShambaGuardEnv(cfg, render_mode="rgb_array" if headless else "human",
                         render_kwargs=dict(show_true_state=args.true_state, cell=cell, fps=args.fps))
    agent = {"greedy": lambda: GreedyAgent(cfg),
             "greedy+fert": lambda: GreedyAgent(cfg, use_fertilizer=True),
             "random": lambda: RandomAgent(args.seed)}[args.agent]()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames, snap_days = [], set(args.snapshots)
    obs, info = env.reset(seed=args.seed)
    if headless:
        frames.append(env.render())
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        if args.verbose:
            print(env.verbose_line())
        if not headless:
            if env._renderer is not None and env._renderer.window_closed:
                break
            continue
        if args.gif and (env.t % args.every == 0 or done):
            frames.append(env.render())
        if env.step_in_day == 0 and env.day in snap_days:
            _save_png(env.render(), outdir / f"{args.agent}_seed{args.seed}_day{env.day:02d}.png")
            snap_days.discard(env.day)
    if headless and args.snapshots:
        _save_png(env.render(), outdir / f"{args.agent}_seed{args.seed}_final.png")

    print(f"Season over: return {info['episode_return']:.1f}, harvest {info['harvest_return']:.1f}, "
          f"harvested {info['plots_harvested']}, lost {info['plots_lost']}, "
          f"stranded {info['stranded']}")
    if args.gif:
        _save_gif(frames, Path(args.gif), fps=args.fps)
        print(f"GIF: {args.gif} ({len(frames)} frames)")
    env.close()


def _save_png(frame, path):
    from PIL import Image
    Image.fromarray(frame).save(path)
    print(f"PNG: {path}")


def _save_gif(frames, path, fps):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


if __name__ == "__main__":
    main()
