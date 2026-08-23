"""
Tier-0 reconstruction probe for the codebook-stitching matrix (zero GPU training).

Answers the *mechanism* half of the stitching ablation without launching a single
RL run: for each cell of the donor matrix it measures how well a frozen stitched
codebook quantizes a held-out batch of the eval game's frames. This previews the
energy-vs-specificity question (does swapping the coarse or the fine level cost
more reconstruction?) before any GPU hours are spent.

Method (matches the paper's D2 quant-error definition in compute_diagnostics.py):
  1. Encode N held-out frames with a FIXED encoder (the eval game's own donor by
     default), giving pre-VQ features z_e in R^256 per spatial position. Holding
     the encoder fixed isolates the *codebook* as the only variable across cells.
  2. For each recipe (per-level donor assignment), cascade-quantize z_e through the
     stitched frozen codebook: at each level assign the nearest code in that
     level's donor embedding, record the residual quant error E^(l), subtract, and
     pass the residual to the next level.
  3. Report per-level E^(l) and total reconstruction error ||z_e - sum_l e_l||^2.

The two homogeneous cells are the references: all-foreign (e.g. [pong,pong,pong])
is the freeze-L012 transfer baseline; all-matched (e.g. [da,da,da]) is the
codebook ceiling. Single-level swaps localise where the foreign codebook hurts.

Usage:
  python scripts/stitch_recon_probe.py \
      --frames results/diag_frames/demon_attack.pt \
      --encoder-ckpt transfer_ckpt/da_seed4_best.ckpt \
      --donors "pong=transfer_ckpt/pong_seed5_best.ckpt,da=transfer_ckpt/da_seed4_best.ckpt"

Frames are the same fixed random-policy batches used for the paper diagnostics
(generate with scripts/gen_diag_frames.py if results/diag_frames/<game>.pt is
absent). Output: results/stitch_recon_probe.csv
"""

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nnet.modules.twister.hrvq.encoder import SpatialHRVQEncoder

NLEVELS = 3
CHUNK = 8192          # spatial positions per assignment chunk (memory bound)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(ckpt_path: str) -> SpatialHRVQEncoder:
    """Load only the encoder subnetwork (CNN + HRVQ) from a full model checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    pref = "encoder_network."
    enc_sd = {k[len(pref):]: v for k, v in sd.items()
              if k.startswith(pref) and not k.startswith("world_model.")}
    enc = SpatialHRVQEncoder()
    missing, _ = enc.load_state_dict(enc_sd, strict=False)
    crit = [m for m in missing if "cnn" in m or "hrvq" in m]
    if crit:
        raise RuntimeError(f"missing critical encoder keys in {ckpt_path}: {crit[:5]}")
    return enc.eval().to(DEVICE)


def load_level_embeddings(ckpt_path: str):
    """Return [emb_L0, emb_L1, emb_L2] codebook tensors (K, D) from a donor checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    embs = []
    for lvl in range(NLEVELS):
        key = f"encoder_network.hrvq.quantizers.{lvl}.embedding"
        if key not in sd:
            raise KeyError(f"{key} not in {ckpt_path}")
        embs.append(sd[key].float().to(DEVICE))
    return embs


@torch.no_grad()
def encode_ze(enc: SpatialHRVQEncoder, frames: torch.Tensor) -> torch.Tensor:
    """frames (N,3,64,64) uint8 -> z_e (N*16, 256) pre-VQ features."""
    chunks = []
    for i in range(0, frames.shape[0], 512):
        x = frames[i:i + 512].to(DEVICE).float() / 255.0 - 0.5   # matches twister.preprocess_inputs
        out = enc(x)
        chunks.append(out["pre_vq_features"].reshape(-1, enc.position_dim).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def cascade_quant_error(ze: torch.Tensor, recipe_embs):
    """Cascade-quantize z_e through a stitched frozen codebook (one emb per level).

    recipe_embs: [emb_L0, emb_L1, emb_L2], each (K, D) for the chosen donor.
    Returns (per_level_E [3], rel_err) where
      E^(l) = mean_positions || r_l - e_{k_l} ||^2          (paper D2 metric), and
      rel_err = mean ||z_e - sum_l e_l||^2 / mean ||z_e||^2  (fraction of feature
      energy the stitched codebook fails to reconstruct; 0 = perfect, lower better).
    """
    per_level = []
    r = ze.to(DEVICE)
    for lvl in range(NLEVELS):
        emb = recipe_embs[lvl]                       # (K, D)
        emb_sq = emb.pow(2).sum(1)                   # (K,)
        err_sum, n = 0.0, 0
        next_r = torch.empty_like(r)
        for i in range(0, r.shape[0], CHUNK):
            rc = r[i:i + CHUNK]                       # (c, D)
            # squared distance to every code, argmin = nearest code (VQ forward)
            d = rc.pow(2).sum(1, keepdim=True) - 2 * rc @ emb.T + emb_sq[None, :]
            idx = d.argmin(1)
            e = emb[idx]
            err_sum += (rc - e).pow(2).sum(-1).sum().item()
            n += rc.shape[0]
            next_r[i:i + CHUNK] = rc - e
        per_level.append(err_sum / n)
        r = next_r
    resid_energy = r.pow(2).sum(-1).mean().item()    # ||z_e - sum_l e_l||^2
    feat_energy = ze.to(DEVICE).pow(2).sum(-1).mean().item()
    rel_err = resid_energy / max(feat_energy, 1e-8)
    return per_level, rel_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="results/diag_frames/demon_attack.pt",
                    help="held-out frame batch (N,3,64,64) uint8 for the eval game")
    ap.add_argument("--encoder-ckpt", default="transfer_ckpt/da_seed4_best.ckpt",
                    help="checkpoint whose encoder produces z_e (default: the matched/eval-game donor)")
    ap.add_argument("--donors", default="pong=transfer_ckpt/pong_seed5_best.ckpt,da=transfer_ckpt/da_seed4_best.ckpt",
                    help="donor name=checkpoint pairs supplying codebook levels")
    ap.add_argument("--out", default="results/stitch_recon_probe.csv")
    args = ap.parse_args()

    donors = {}
    for pair in args.donors.split(","):
        name, _, path = pair.partition("=")
        donors[name.strip()] = path.strip()
    names = sorted(donors)
    if len(names) != 2:
        print(f"[note] {len(names)} donors given; the full 2^3 matrix assumes exactly 2.")

    frames_path = Path(args.frames)
    if not frames_path.is_file():
        sys.exit(f"frames not found: {frames_path}\n"
                 f"Generate the held-out batch first: python scripts/gen_diag_frames.py")
    frames = torch.load(frames_path)
    print(f"device={DEVICE}  frames={tuple(frames.shape)}  encoder={args.encoder_ckpt}")

    enc = load_encoder(args.encoder_ckpt)
    ze = encode_ze(enc, frames)
    print(f"z_e: {tuple(ze.shape)}  donors: {donors}\n")

    # Per-donor level embeddings, loaded once.
    donor_embs = {name: load_level_embeddings(path) for name, path in donors.items()}

    # Full factorial over donors at each level: |donors|^3 recipes.
    recipes = list(product(names, repeat=NLEVELS))
    rows = []
    hdr = f"{'recipe':<20} {'E0':>9} {'E1':>9} {'E2':>9} {'rel_err':>9}"
    print(hdr); print("-" * len(hdr))
    for recipe in recipes:
        embs = [donor_embs[recipe[l]][l] for l in range(NLEVELS)]
        per_level, rel_err = cascade_quant_error(ze, embs)
        tag = "[" + ",".join(recipe) + "]"
        print(f"{tag:<20} {per_level[0]:>9.2f} {per_level[1]:>9.2f} {per_level[2]:>9.2f} {rel_err:>9.4f}")
        rows.append({"recipe": tag, "L0": recipe[0], "L1": recipe[1], "L2": recipe[2],
                     "E0": per_level[0], "E1": per_level[1], "E2": per_level[2], "rel_err": rel_err})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recipe", "L0", "L1", "L2", "E0", "E1", "E2", "rel_err"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} recipes -> {out}")


if __name__ == "__main__":
    main()
