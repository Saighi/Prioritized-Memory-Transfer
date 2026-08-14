"""Render the PFC -> hippocampus schematic: fill the Jinja2 template from spec_bio.py,
compile to PDF, convert to SVG.

    python render.py                       # all four propositions
    python render.py --variant b           # just one (a|b|c|d or its long name)
    python render.py --no-svg

Needs: a LaTeX toolchain with tikz+standalone (MiKTeX/TeX Live) and, for SVG,
`dvisvgm` (ships with both). Outputs pfc_hpc_<v>_<name>.pdf (+ .svg) next to this file.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so `import spec_bio` works
import spec_bio                                # noqa: E402

from jinja2 import Template                    # noqa: E402

SX, SY = 1.0, 1.0                              # spec_bio is already in cm


def render(variant: str, tpl: Template, want_svg: bool) -> None:
    sc = spec_bio.build_scene(variant)
    stem = f"pfc_hpc_{sc.meta['variant']}_{sc.meta['name']}"
    tex = tpl.render(boxes=sc.boxes, links=sc.links, texts=sc.texts, rules=sc.rules,
                     legend=sc.legend, colors=spec_bio.COLORS, sx=SX, sy=SY)
    (HERE / f"{stem}.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {stem}.tex")

    engine = shutil.which("pdflatex") or shutil.which("lualatex")
    if not engine:
        print("  no pdflatex/lualatex on PATH — .tex only.")
        return
    for _ in range(2):                         # twice so node positions settle
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                        f"{stem}.tex"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    print(f"  compiled {stem}.pdf")

    if want_svg:
        dvisvgm = shutil.which("dvisvgm")
        if dvisvgm:
            subprocess.run([dvisvgm, "--pdf", f"{stem}.pdf", "-o", f"{stem}.svg"],
                           cwd=HERE, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            print(f"  converted {stem}.svg")
        else:
            print("  dvisvgm not found — PDF only.")

    for ext in ("aux", "log"):                 # tidy; the .tex is kept, as in tikz/
        (HERE / f"{stem}.{ext}").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    help="a|b|c|d, their long names, or 'all' (default)")
    ap.add_argument("--no-svg", action="store_true")
    args = ap.parse_args()

    tpl = Template((HERE / "pfc_hpc.tex.j2").read_text(encoding="utf-8"))
    variants = spec_bio.VARIANTS if args.variant == "all" else (args.variant,)
    for v in variants:
        render(v, tpl, not args.no_svg)


if __name__ == "__main__":
    main()
