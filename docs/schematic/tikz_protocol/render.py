"""Render the training-protocol schematic: fill the Jinja2 template from spec_protocol.py,
compile to PDF, convert to SVG.

    python render.py [--no-svg]

Same toolchain as the sibling schematics (LaTeX with tikz+standalone; dvisvgm for the
SVG). Outputs protocol_ab.pdf (+ protocol_ab.svg) next to this file.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so `import spec_protocol` works
import spec_protocol                           # noqa: E402

from jinja2 import Template                     # noqa: E402

SX, SY = 1.6, 1.6                               # cm/grid-unit
STEM = "protocol_ab"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-svg", action="store_true")
    args = ap.parse_args()

    sc = spec_protocol.build_scene()
    geom = {"r_ring": spec_protocol.R_RING, "panel_dx": spec_protocol.PANEL_DX,
            "cy_s": spec_protocol.CY_S, "cy_t": spec_protocol.CY_T}
    tpl = Template((HERE / "protocol.tex.j2").read_text(encoding="utf-8"))
    tex = tpl.render(panels=sc["panels"], colors=spec_protocol.COLORS,
                     opacity_off=spec_protocol.OPACITY_OFF, geom=geom, sx=SX, sy=SY)
    (HERE / f"{STEM}.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {STEM}.tex")

    engine = shutil.which("pdflatex") or shutil.which("lualatex")
    if not engine:
        print("no pdflatex/lualatex on PATH — wrote .tex only.")
        return
    for _ in range(2):                          # twice so node positions settle
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                        f"{STEM}.tex"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    print(f"compiled {STEM}.pdf")

    if not args.no_svg:
        dvisvgm = shutil.which("dvisvgm")
        if dvisvgm:
            subprocess.run([dvisvgm, "--pdf", f"{STEM}.pdf", "-o", f"{STEM}.svg"],
                           cwd=HERE, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            print(f"converted {STEM}.svg")
        else:
            print("dvisvgm not found — PDF only.")

    for ext in ("aux", "log"):                  # tidy
        (HERE / f"{STEM}.{ext}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
