"""
Render a TWISTER-style game gallery: one representative observation per game in
the study, laid out to motivate the source -> target domain-distance axis of
Section 4.1 / Table 1.

Frames come from the fixed random-policy held-out batches produced by
scripts/gen_diag_frames.py (results/diag_frames/<game>.pt, uint8 (N,3,64,64)).
For each game we pick the highest-variance frame from an early window, which
reliably avoids the blank inter-wave / serve-gap frames (Pong serve, Demon
Attack empty waves) that a fixed index would sometimes land on.

Output: results/figures/game_gallery.{pdf,png}

Usage:
  python scripts/plot_game_gallery.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FRAME_DIR = ROOT / "results" / "diag_frames"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (game key, display name, role, unique RGB, BG dominance, object categories)
# Metrics are Table 1 (tab:game_properties); order follows Table 1: source first,
# then targets by decreasing background dominance (the domain-distance axis).
GAMES = [
    ("pong",         "Pong",         "Source", 5,   0.862, 3),
    ("demon_attack", "Demon Attack", "Target", 35,  0.890, 5),
    ("freeway",      "Freeway",      "Target", 71,  0.733, 2),
    ("breakout",     "Breakout",     "Target", 9,   0.685, 3),
    ("ms_pacman",    "Ms Pac-Man",   "Target", 11,  0.522, 5),
    ("up_n_down",    "Up'n'Down",    "Target", 127, 0.371, 4),
]

# Sample frames from this many leading frames; pick the most "active" one.
WINDOW = 400


def pick_frame(frames: torch.Tensor) -> np.ndarray:
    """Return an (H,W,3) uint8 frame: the highest spatial-variance frame in the
    leading window (most on-screen content, avoids blank frames)."""
    w = frames[: min(WINDOW, frames.shape[0])].float()
    # per-frame spatial std over channels+pixels
    var = w.reshape(w.shape[0], -1).std(dim=1)
    idx = int(torch.argmax(var).item())
    img = frames[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img


def main():
    missing = [g for g, *_ in GAMES if not (FRAME_DIR / f"{g}.pt").exists()]
    if missing:
        sys.exit(
            f"Missing frame batches for {missing}. Run:\n"
            f"  python scripts/gen_diag_frames.py --games {' '.join(missing)}"
        )

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.1, nrows * 2.5)
    )
    axes = axes.ravel()

    for ax, (key, name, role, rgb, bg, obj) in zip(axes, GAMES):
        frames = torch.load(FRAME_DIR / f"{key}.pt")
        img = pick_frame(frames)
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        # Source vs target border colour
        edge = "#c0392b" if role == "Source" else "#2c3e50"
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(2.2 if role == "Source" else 1.2)
        title = f"{name}" + (r"  $\bigstar$" if role == "Source" else "")
        ax.set_title(title, fontsize=11, pad=4,
                     color=edge, fontweight="bold")
        ax.set_xlabel(
            f"{role}\nBG {bg:.2f} · {obj} obj · {rgb} RGB",
            fontsize=8, labelpad=3,
        )

    for ax in axes[len(GAMES):]:
        ax.axis("off")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"game_gallery.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[ok] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
