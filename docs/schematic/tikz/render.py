"""Render the TikZ schematic: fill the Jinja2 template from spec.py, compile to PDF,
and convert to SVG.

    python render.py [--phase sleep|wake] [--no-svg]

Needs: a LaTeX toolchain with tikz+standalone (MiKTeX/TeX Live) and, for SVG,
`dvisvgm` (ships with both). Outputs network.pdf (+ network.svg) next to this file.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so `import spec` works
import spec                                     # noqa: E402

from jinja2 import Template                      # noqa: E402

SX, SY, BEND = 1.7, 1.6, 14                      # cm/unit, cm/unit, bend degrees


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sleep", choices=["sleep", "wake"])
    ap.add_argument("--no-svg", action="store_true")
    args = ap.parse_args()

    sp = spec.build_spec(phase=args.phase)
    tpl = Template((HERE / "network.tex.j2").read_text(encoding="utf-8"))
    tex = tpl.render(nodes=sp["nodes"], edges=sp["edges"],
                     note_tex=sp["meta"]["note_tex"], sx=SX, sy=SY, bend=BEND)
    tex_path = HERE / "network.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"wrote {tex_path.name}")

    engine = shutil.which("pdflatex") or shutil.which("lualatex")
    if not engine:
        print("no pdflatex/lualatex on PATH — wrote .tex only.")
        return
    for _ in range(2):                            # twice so node positions settle
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                        "network.tex"], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL)
    print("compiled network.pdf")

    if not args.no_svg:
        dvisvgm = shutil.which("dvisvgm")
        if dvisvgm:
            subprocess.run([dvisvgm, "--pdf", "network.pdf", "-o", "network.svg"],
                           cwd=HERE, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            print("converted network.svg")
        else:
            print("dvisvgm not found — PDF only.")

    for ext in ("aux", "log"):                    # tidy
        (HERE / f"network.{ext}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
