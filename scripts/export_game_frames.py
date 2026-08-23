"""
Export one clean representative observation per game as a standalone PNG, for the
TikZ game-gallery figure (Figure~\ref{fig:game_gallery}).

Each frame is the highest spatial-variance frame from the leading window of the
fixed random-policy held-out batch (results/diag_frames/<game>.pt), nearest-
upscaled 64 -> 256 so the Atari pixels stay crisp when TikZ rescales them.

Output (flat, matching the figures/ build convention):
  results/figures/game_<key>.png

Usage:
  python scripts/export_game_frames.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRAME_DIR = ROOT / "results" / "diag_frames"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAMES = ["pong", "demon_attack", "freeway", "breakout", "ms_pacman", "up_n_down",
         "asterix"]
WINDOW = 400      # pick the most active frame from this leading window
UPSCALE = 256     # nearest-neighbour target size


def pick_frame(frames: torch.Tensor) -> np.ndarray:
    w = frames[: min(WINDOW, frames.shape[0])].float()
    var = w.reshape(w.shape[0], -1).std(dim=1)
    idx = int(torch.argmax(var).item())
    return frames[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H,W,3)


def main():
    missing = [g for g in GAMES if not (FRAME_DIR / f"{g}.pt").exists()]
    if missing:
        sys.exit(
            f"Missing frame batches for {missing}. Run:\n"
            f"  python scripts/gen_diag_frames.py --games {' '.join(missing)}"
        )
    for g in GAMES:
        img = pick_frame(torch.load(FRAME_DIR / f"{g}.pt"))
        im = Image.fromarray(img, mode="RGB").resize(
            (UPSCALE, UPSCALE), Image.NEAREST
        )
        out = OUT_DIR / f"game_{g}.png"
        im.save(out)
        print(f"[ok] {out}")


if __name__ == "__main__":
    main()
