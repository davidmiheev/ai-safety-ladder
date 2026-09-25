"""Render the section write-ups (writeups/*.typ) to PDF.

  uv run python scripts/build_writeups.py            # refresh figures/data from outputs/, build all
  uv run python scripts/build_writeups.py 02         # only sections whose file name starts with 02
  uv run python scripts/build_writeups.py --preview  # also dump page PNGs for a quick visual check

Experiment scripts write figures and JSON into outputs/ (git-ignored). The write-ups
must stay renderable from a fresh clone, so the files they use are copied into
writeups/figures and writeups/data (committed). ARTIFACTS lists them; a missing
source keeps the committed copy and prints a warning, so you can rebuild the PDFs
without re-running every experiment.
"""

import argparse
import shutil
import sys
from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parents[1]
WRITEUPS = ROOT / "writeups"
OUTPUTS = ROOT / "outputs"

# outputs/<path>  ->  writeups/<figures|data>/<basename>
ARTIFACTS = [
    "00_setup/logit_lens_gpt2-small.png",
    "00_setup/first_pass_gpt2-small.json",
    "00_setup/nnsight_gpt2.json",
    "00_setup/sae_gpt2-small-res-jb_L8.png",
    "00_setup/sae_gpt2-small-res-jb_L8.json",
    "01_mech_interp/induction_heads.png",
    "01_mech_interp/induction_where_vs_what.png",
    "01_mech_interp/induction_heads.json",
]


def sync_artifacts() -> None:
    for rel in ARTIFACTS:
        src = OUTPUTS / rel
        sub = "data" if src.suffix == ".json" else "figures"
        dst = WRITEUPS / sub / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists():
            print(f"  warning: {rel} not in outputs/, keeping committed copy", file=sys.stderr)
        else:
            raise SystemExit(f"missing {rel}: run the experiment that produces it first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="*", help="section prefixes to build (default: all)")
    parser.add_argument("--preview", action="store_true", help="also write page PNGs to outputs/writeup_preview/")
    args = parser.parse_args()

    sync_artifacts()
    sources = sorted(p for p in WRITEUPS.glob("[0-9]*.typ"))
    if args.only:
        sources = [p for p in sources if any(p.name.startswith(o) for o in args.only)]
    (WRITEUPS / "pdf").mkdir(exist_ok=True)
    for src in sources:
        pdf = WRITEUPS / "pdf" / f"{src.stem}.pdf"
        typst.compile(str(src), output=str(pdf), root=str(WRITEUPS))
        print(f"{src.relative_to(ROOT)} -> {pdf.relative_to(ROOT)} ({pdf.stat().st_size // 1024} KB)")
        if args.preview:
            prev = OUTPUTS / "writeup_preview"
            prev.mkdir(parents=True, exist_ok=True)
            typst.compile(str(src), output=str(prev / f"{src.stem}-{{p}}.png"), root=str(WRITEUPS), ppi=80)


if __name__ == "__main__":
    main()
