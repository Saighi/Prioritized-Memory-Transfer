"""Render the D2 schematic: fill the Jinja2 template from spec.py, then (if the `d2`
CLI is installed) compile to SVG.

    python render.py [--phase sleep|wake] [--layout dagre|elk|tala]

D2 is not a Python package — install the CLI once:
    winget install terrastruct.d2      # Windows
    brew install d2                    # macOS
    # or: https://d2lang.com/tour/install
Then this script produces network.d2 (+ network.svg). Without the CLI it writes the
.d2 source and prints the one command to render it (or paste into https://play.d2lang.com).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import spec                                       # noqa: E402

from jinja2 import Template                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sleep", choices=["sleep", "wake"])
    ap.add_argument("--layout", default="elk", choices=["dagre", "elk", "tala"])
    args = ap.parse_args()

    sp = spec.build_spec(phase=args.phase)
    tpl = Template((HERE / "network.d2.j2").read_text(encoding="utf-8"))
    d2 = tpl.render(nodes=sp["nodes"], edges=sp["edges"], colors=spec.COLORS)
    d2_path = HERE / "network.d2"
    d2_path.write_text(d2, encoding="utf-8")
    print(f"wrote {d2_path.name}")

    exe = shutil.which("d2")
    if exe:
        subprocess.run([exe, "--layout", args.layout, "network.d2", "network.svg"],
                       cwd=HERE, check=True)
        print("rendered network.svg")
    else:
        print("d2 CLI not found. Render with:")
        print(f"    d2 --layout {args.layout} {d2_path.name} network.svg")
        print("or paste network.d2 into https://play.d2lang.com")


if __name__ == "__main__":
    main()
