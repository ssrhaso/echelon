"""Qualitative reconstruction grids from saved checkpoints.

For a checkpoint we load the encoder (CNN + 3-level HRVQ) and the decoder's
transposed CNN, encode a fixed held-out frame batch for that game, quantize
through all three RVQ levels, and decode straight from the summed spatial codes
via spatial_cascade_decode (which bypasses the decoder's flat projection - the
spatial 16x256 tokens reshape directly to the decoder's 256x4x4 input). The
result is the image the codebook can reconstruct *without* the world model, so
it isolates what the tokenizer preserves on the target distribution.

The output grid pairs each sampled original frame with its reconstruction,
giving a visual read on the quantization-error diagnostic (Table: results_
diagnostics): conditions whose codebook fits the target (low E^(l)) reconstruct
recognisable frames; the fully-frozen conditions, which carry two-to-three
orders of magnitude more quant error, smear into the wrong Pong cells.

Output: results/recon/<game>_<condition>_seed<seed>.png

Usage:
  python scripts/gen_reconstructions.py                      # all local ckpts
  python scripts/gen_reconstructions.py --games demon_attack # one game
  python scripts/gen_reconstructions.py --conditions scratch freeze-all --n 6
"""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnet.modules.twister.hrvq.encoder import SpatialHRVQEncoder
from nnet.modules.twister.decoder_network import DecoderNetwork
from nnet.modules.twister.hrvq.decoder import spatial_cascade_decode

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints_local"
FRAME_DIR = ROOT / "results" / "diag_frames"
OUT_DIR = ROOT / "results" / "recon"

NLEVELS = 3
# Decoder proj is unused on the spatial cascade path, but feat_size must match
# the checkpoint tensor shape so the state_dict loads cleanly.
DEC_FEAT_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIRNAME_RE = re.compile(
    r"^([a-z_]+)_(scratch|warmstart|freeze-L0|freeze-L01|freeze-L012|freeze-enc|freeze-all)_seed(\d+)$")


def parse_dirname(name):
    m = DIRNAME_RE.match(name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def find_ckpt(folder):
    cks = sorted(folder.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no .ckpt in {folder}")
    return cks[0]


def load_models(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]

    def sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    enc = SpatialHRVQEncoder()
    enc.load_state_dict(sub("encoder_network."), strict=False)
    dec = DecoderNetwork(feat_size=DEC_FEAT_SIZE)
    missing, _ = dec.load_state_dict(sub("decoder_network."), strict=False)
    crit = [m for m in missing if "cnn" in m]
    if crit:
        raise RuntimeError(f"missing decoder CNN keys in {ckpt_path}: {crit[:5]}")
    return enc.eval().to(DEVICE), dec.eval().to(DEVICE)


def to_display(x):
    """Model space ([-0.5, 0.5]) -> (H, W, 3) float in [0, 1] for imshow."""
    img = (x.detach().cpu().float() + 0.5).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def reconstruct(enc, dec, frames):
    """frames: (S, 3, 64, 64) uint8 -> full reconstruction (S, 3, 64, 64) float."""
    x = frames.to(DEVICE).float() / 255.0 - 0.5
    z_q_levels = enc(x)["hrvq_info"]["z_q_levels_spatial"]
    return spatial_cascade_decode(dec, z_q_levels, up_to_level=NLEVELS - 1).mode()


@torch.no_grad()
def reconstruct_cascade(enc, dec, frames):
    """Per-level partial reconstructions: list of (S, 3, 64, 64), one per level
    using codes 0..l. Reveals the coarse-to-fine refinement of the residual
    quantizer: L0 carries the dominant variation, later levels add residual
    detail."""
    x = frames.to(DEVICE).float() / 255.0 - 0.5
    z_q_levels = enc(x)["hrvq_info"]["z_q_levels_spatial"]
    return [spatial_cascade_decode(dec, z_q_levels, up_to_level=l).mode()
            for l in range(NLEVELS)]


def render(frames, recon, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = frames.shape[0]
    fig, axes = plt.subplots(2, s, figsize=(1.4 * s, 3.0))
    if s == 1:
        axes = axes.reshape(2, 1)
    for j in range(s):
        axes[0, j].imshow(to_display(frames[j].float() / 255.0 - 0.5))
        axes[1, j].imshow(to_display(recon[j]))
        for i in range(2):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("reconstruction", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_cascade(frames, recons, out_path, title):
    """Columns: original | L0 | L0+1 | L0+1+2 ; one row per sampled frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = frames.shape[0]
    col_titles = ["original"] + [f"L0..{l}" if l else "L0" for l in range(NLEVELS)]
    ncol = len(col_titles)
    fig, axes = plt.subplots(s, ncol, figsize=(1.5 * ncol, 1.5 * s))
    axes = axes.reshape(s, ncol)
    for i in range(s):
        axes[i, 0].imshow(to_display(frames[i].float() / 255.0 - 0.5))
        for l in range(NLEVELS):
            axes[i, l + 1].imshow(to_display(recons[l][i]))
        for j in range(ncol):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None, help="filter games (default: all found)")
    ap.add_argument("--conditions", nargs="*", default=None, help="filter conditions")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="filter seeds")
    ap.add_argument("--n", type=int, default=8, help="frames per grid")
    ap.add_argument("--stride", type=int, default=257, help="spacing between sampled frames")
    ap.add_argument("--mode", choices=["pair", "cascade", "both"], default="pair",
                    help="pair: original/reconstruction; cascade: per-level coarse-to-fine")
    args = ap.parse_args()

    print(f"device={DEVICE}")
    frame_cache = {}
    dirs = sorted(d for d in CKPT_DIR.iterdir() if d.is_dir() and parse_dirname(d.name))
    n_done = 0
    for d in dirs:
        game, cond, seed = parse_dirname(d.name)
        if args.games and game not in args.games:
            continue
        if args.conditions and cond not in args.conditions:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        frame_file = FRAME_DIR / f"{game}.pt"
        if not frame_file.exists():
            print(f"[skip] {d.name}: no frames {frame_file.name}")
            continue
        try:
            enc, dec = load_models(find_ckpt(d))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[skip] {d.name}: {e}")
            continue
        if game not in frame_cache:
            frame_cache[game] = torch.load(frame_file)
        all_frames = frame_cache[game]
        idx = torch.arange(args.n) * args.stride % all_frames.shape[0]
        frames = all_frames[idx]
        tag = f"{game}  {cond}  seed{seed}"
        if args.mode in ("pair", "both"):
            out = OUT_DIR / f"{game}_{cond}_seed{seed}.png"
            render(frames, reconstruct(enc, dec, frames), out, tag)
            print(f"[ok] {out.relative_to(ROOT)}")
        if args.mode in ("cascade", "both"):
            out = OUT_DIR / f"{game}_{cond}_seed{seed}_cascade.png"
            render_cascade(frames[:4], reconstruct_cascade(enc, dec, frames[:4]), out, tag)
            print(f"[ok] {out.relative_to(ROOT)}")
        n_done += 1
    print(f"\nWrote {n_done} reconstruction grids -> {OUT_DIR}")


if __name__ == "__main__":
    main()
