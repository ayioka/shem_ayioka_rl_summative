"""Zip the code Colab needs: python scripts/pack_colab.py -> dist/shambaguard_colab.zip

Includes environment/, training/, scripts/, configs/, tests/, pyproject.toml and the greedy
clone models/bc/bc_greedy.pt (the warm start). Logs, other models and the venv stay out.
Upload the zip to Google Drive under MyDrive/ShambaGuard/ and open
notebooks/colab_sweeps.ipynb in Colab.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("environment", "training", "scripts", "configs", "tests")
FILES = ("pyproject.toml", "models/bc/bc_greedy.pt", "models/bc/bc_greedy.json")


def main():
    out = ROOT / "dist" / "shambaguard_colab.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
        for f in FILES:
            p = ROOT / f
            if not p.exists():
                raise SystemExit(f"missing {f} - run python training/bc.py first")
            z.write(p, f)
            n += 1
    print(f"wrote {out.relative_to(ROOT)} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
