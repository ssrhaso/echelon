"""Spatial hit rate of the shared core level-0 codes, emitted as a CSV.

Same checkpoints, frame batches and core code set as analysis_pta/core_vocab.py;
this writes the numbers behind the spatial map so the paper can draw it in TikZ
instead of embedding a raster.

Output: analysis_pta/core_vocab_spatial.csv, one row per (game, row, col) of the
4x4 token grid, with the fraction of frames whose token at that position is one
of the core codes.

Usage: python analysis_pta/core_vocab_spatial.py
"""

from pathlib import Path
import csv
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis_pta.core_vocab import (  # noqa: E402
    CKPT_DIR, FRAME_DIR, OUT_DIR, GAMES, find_ckpt, load_encoder, level0_indices,
)


def main():
    core = [int(r["code"]) for r in csv.DictReader(open(OUT_DIR / "core_vocab_codes.csv"))]
    print("core codes:", core)

    idx = {}
    src_enc = load_encoder(find_ckpt(CKPT_DIR / "pong_source"))
    idx["pong"] = level0_indices(src_enc, torch.load(FRAME_DIR / "pong.pt"))
    for g in GAMES:
        enc = load_encoder(find_ckpt(CKPT_DIR / f"{g}_freeze-L0_seed0"))
        idx[g] = level0_indices(enc, torch.load(FRAME_DIR / f"{g}.pt"))
        print("encoded", g)

    rows = []
    for g, ids in idx.items():
        hit = torch.zeros_like(ids, dtype=torch.bool)
        for c in core:
            hit |= ids == c
        heat = hit.float().mean(0).reshape(4, 4)
        for r in range(4):
            for c in range(4):
                rows.append({"game": g, "row": r, "col": c,
                             "hit_rate": round(float(heat[r, c]), 4)})
        print(g, "overall", round(float(hit.float().mean()), 4))

    out = OUT_DIR / "core_vocab_spatial.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "row", "col", "hit_rate"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
