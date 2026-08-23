"""
Per-level transfer diagnostics (D1/D2/D3) over saved checkpoints.

For each target checkpoint we load ONLY the encoder subnetwork
(`encoder_network.*` = CNN + 3-level HRVQ codebooks), encode a fixed held-out
frame batch for that game, and compute:

  D1  Jaccard J^(l)   : overlap of the target active code set with the Pong
                        source active set, per RVQ level.

                        J^(l) is confounded by active-set size: |A_tgt| ranges
                        over ~50-350 across conditions, and J falls with |A_tgt|
                        even under pure chance overlap. We therefore also record
                        the raw ingredients (|A_src|, |A_tgt|, intersection,
                        union) so a size-matched null can be computed downstream
                        (see scripts/aggregate_diagnostics.py).
  D2  Quant error E^(l): mean squared residual quant error per level on the
                        target frames (||r_l - e_{k_l}||^2 averaged over 16N
                        spatial positions).

                        E^(l) is an *unnormalised* MSE in each checkpoint's own
                        pre-VQ feature space, so it is NOT comparable across
                        conditions on its own: two encoders that differ only by a
                        scale factor on z produce E^(l) differing by its square.
                        We therefore also record the pre-VQ feature energy
                        zenergy = E||z||^2 for every checkpoint. Because
                        E^(l) = E||r^(l+1)||^2 by construction, (zenergy, E0, E1,
                        E2) are exactly the residual energies R^(0..3), which
                        makes the scale-free quantities derivable downstream:
                          Erel^(l) = E^(l) / R^(0)   (fraction of feature energy
                                                      left after level l)
                          contraction^(l) = R^(l+1) / R^(l)  (per-level work done)
  D3  Encoder drift rho_enc : mean cosine similarity between the source-encoder
                        pre-VQ features and this checkpoint's pre-VQ features on
                        the SAME target frames. 1.00 by construction when the
                        encoder is frozen+transferred (freeze-enc / freeze-all).

Also reports per-level perplexity P^(l) and active-code count |A^(l)| on the
fixed batch.

Active set: A^(l) = { k : p_k > tau },  tau = 1/(2K) = 1/1024  (K=512).

The Pong source active set A_Pong and the per-game source-encoder features are
computed ONCE from checkpoints_local/pong_source.

Output: results/diagnostics_raw.csv  (one row per game/condition/seed)

Usage:
  python scripts/compute_diagnostics.py
"""

from pathlib import Path
import re
import csv
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnet.modules.twister.hrvq.encoder import SpatialHRVQEncoder

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints_local"
FRAME_DIR = ROOT / "results" / "diag_frames"
OUT_CSV = ROOT / "results" / "diagnostics_raw.csv"

K = 512
TAU = 1.0 / (2 * K)
NLEVELS = 3
BS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]
SOURCE_GAME = "pong"


def find_ckpt(folder: Path) -> Path:
    cks = sorted(folder.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no .ckpt in {folder}")
    return cks[0]


def load_encoder(ckpt_path: Path) -> SpatialHRVQEncoder:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    pref = "encoder_network."
    enc_sd = {k[len(pref):]: v for k, v in sd.items()
              if k.startswith(pref) and not k.startswith("world_model.")}
    enc = SpatialHRVQEncoder()
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    # Only tolerate absence of nothing critical; report anomalies.
    crit = [m for m in missing if "cnn" in m or "hrvq" in m]
    if crit:
        raise RuntimeError(f"missing critical keys in {ckpt_path}: {crit[:5]}")
    enc.eval().to(DEVICE)
    return enc


@torch.no_grad()
def encode(enc: SpatialHRVQEncoder, frames: torch.Tensor):
    """frames: (N,3,64,64) uint8 -> (idx[level] (M,) long, ze (M,256) float). M=N*16."""
    idx_levels = [[] for _ in range(NLEVELS)]
    ze_chunks = []
    for i in range(0, frames.shape[0], BS):
        x = frames[i:i + BS].to(DEVICE).float() / 255.0 - 0.5   # matches twister.preprocess_inputs
        out = enc(x)
        info = out["hrvq_info"]
        for l in range(NLEVELS):
            idx_levels[l].append(info["indices"][l].reshape(-1).cpu())
        ze = out["pre_vq_features"].reshape(-1, enc.position_dim).cpu()  # (B*16,256)
        ze_chunks.append(ze)
    idx = [torch.cat(c) for c in idx_levels]
    ze = torch.cat(ze_chunks)
    return idx, ze


def active_mask(idx_level: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(idx_level, minlength=K).float()
    probs = counts / counts.sum()
    return probs > TAU, probs


def perplexity(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def quant_errors(ze: torch.Tensor, idx, embeddings):
    """Per-level mean squared residual quant error on the batch."""
    errs = []
    r = ze
    for l in range(NLEVELS):
        e = embeddings[l][idx[l]]               # (M,256)
        errs.append(float(((r - e) ** 2).sum(-1).mean()))
        r = r - e
    return errs


def cosine_drift(ze_src: torch.Tensor, ze_tgt: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(ze_src, dim=-1)
    b = torch.nn.functional.normalize(ze_tgt, dim=-1)
    return float((a * b).sum(-1).mean())


def parse_dirname(name: str):
    m = re.match(r"^([a-z_]+)_(scratch|warmstart|freeze-L0|freeze-L01|freeze-L012|freeze-enc|freeze-all)_seed(\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def main():
    frames = {g: torch.load(FRAME_DIR / f"{g}.pt") for g in GAMES + [SOURCE_GAME]}
    print(f"device={DEVICE}; frames: " + ", ".join(f"{g}={tuple(frames[g].shape)}" for g in frames))

    # --- Source reference (computed once) ---
    src_enc = load_encoder(find_ckpt(CKPT_DIR / "pong_source"))
    src_idx_pong, _ = encode(src_enc, frames[SOURCE_GAME])
    src_active = [active_mask(src_idx_pong[l])[0] for l in range(NLEVELS)]  # A_Pong masks
    print("A_Pong sizes:", [int(m.sum()) for m in src_active])
    # source-encoder pre-VQ features on each TARGET game (for D3)
    src_ze_game = {g: encode(src_enc, frames[g])[1] for g in GAMES}

    rows = []
    dirs = sorted(d for d in CKPT_DIR.iterdir() if d.is_dir() and parse_dirname(d.name))
    for d in dirs:
        game, cond, seed = parse_dirname(d.name)
        if game not in GAMES:
            continue
        try:
            ckpt = find_ckpt(d)
        except FileNotFoundError:
            print(f"[skip] {d.name}: no .ckpt"); continue
        enc = load_encoder(ckpt)
        idx, ze = encode(enc, frames[game])
        embeddings = [enc.hrvq.quantizers[l].embedding.cpu() for l in range(NLEVELS)]

        row = {"game": game, "condition": cond, "seed": seed}
        for l in range(NLEVELS):
            tgt_mask, probs = active_mask(idx[l])
            inter = int((src_active[l] & tgt_mask).sum())
            union = int((src_active[l] | tgt_mask).sum())
            row[f"J{l}"] = inter / union if union else float("nan")
            row[f"perp{l}"] = perplexity(probs)
            row[f"active{l}"] = int(tgt_mask.sum())
            # raw ingredients for the size-matched null (see module docstring)
            row[f"srcactive{l}"] = int(src_active[l].sum())
            row[f"inter{l}"] = inter
            row[f"union{l}"] = union
        for l, e in enumerate(quant_errors(ze, idx, embeddings)):
            row[f"E{l}"] = e
        # pre-VQ feature energy R^(0) = E||z||^2: the scale that makes E^(l)
        # comparable across checkpoints with different encoders.
        row["zenergy"] = float((ze ** 2).sum(-1).mean())
        row["rho_enc"] = cosine_drift(src_ze_game[game], ze)
        rows.append(row)
        print(f"[ok] {d.name:34s} J={[round(row[f'J{l}'],3) for l in range(3)]} "
              f"E={[round(row[f'E{l}'],2) for l in range(3)]} "
              f"Erel={[round(row[f'E{l}']/row['zenergy'],3) for l in range(3)]} "
              f"|z|^2={row['zenergy']:.1f} rho={row['rho_enc']:.3f}")

    cols = ["game", "condition", "seed",
            "J0", "J1", "J2", "E0", "E1", "E2", "rho_enc",
            "perp0", "perp1", "perp2", "active0", "active1", "active2",
            "srcactive0", "srcactive1", "srcactive2",
            "inter0", "inter1", "inter2", "union0", "union1", "union2",
            "zenergy"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
