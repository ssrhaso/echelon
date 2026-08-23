"""Cross-game consistency of the level-0 codes adopted under freeze-CB0.

Under freeze-CB0 the level-0 table is pinned at the Pong source in every run,
so code indices are comparable across games and seeds. This script asks which
source codes each target adopts, whether the three targets adopt the same
codes, and what characterizes the shared set.

Outputs:
  analysis_pta/core_vocab_per_seed.csv   per run adoption statistics
  analysis_pta/core_vocab_pairs.csv      pairwise and triple consistency
  analysis_pta/core_vocab_codes.csv      the shared codes with usage and norm
  figures/core_vocab_patches.png         patch montage (with --figures)
  figures/core_vocab_spatial.png         spatial hit rates (with --figures)

Null models: adoption of source-active codes uses the hypergeometric null over
all K codes (as in the paper's reuse diagnostic). Cross-game consistency uses
the stricter conditional null of random subsets drawn within the source-active
set, so it measures agreement beyond the shared bias toward source codes.

Usage: python analysis_pta/core_vocab.py [--figures]
"""

from pathlib import Path
import argparse
import csv
import itertools
import json
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nnet.modules.twister.hrvq.encoder import SpatialHRVQEncoder

CKPT_DIR = ROOT / "checkpoints_local"
FRAME_DIR = ROOT / "results" / "diag_frames"
OUT_DIR = ROOT / "analysis_pta"
FIG_DIR = ROOT / "figures"

K = 512
TAU = 1.0 / (2 * K)
BS = 512
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]
SEEDS = [0, 1, 2]
N_PATCH = 6


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
    enc.load_state_dict(enc_sd, strict=False)
    enc.eval()
    return enc


@torch.no_grad()
def level0_indices(enc: SpatialHRVQEncoder, frames: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(0, frames.shape[0], BS):
        x = frames[i:i + BS].float() / 255.0 - 0.5
        idx0 = enc(x)["hrvq_info"]["indices"][0]
        out.append(idx0.reshape(x.shape[0], -1).cpu())
    return torch.cat(out)


def usage(idx: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(idx.reshape(-1), minlength=K).float()
    return counts / counts.sum()


def hyper_z(inter: int, s: int, t: int, k: int):
    m = s * t / k
    var = s * t * (k - s) * (k - t) / (k * k * (k - 1))
    return (inter - m) / var ** 0.5, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    frames = {g: torch.load(FRAME_DIR / f"{g}.pt") for g in GAMES + ["pong"]}

    src_enc = load_encoder(find_ckpt(CKPT_DIR / "pong_source"))
    emb0 = src_enc.hrvq.quantizers[0].embedding.detach()
    norms = emb0.norm(dim=-1)
    src_idx = level0_indices(src_enc, frames["pong"])
    p_src = usage(src_idx)
    src_active = p_src > TAU
    s_src = int(src_active.sum())
    src_rank = torch.argsort(torch.argsort(p_src)).float() / (K - 1)

    per_seed_rows = []
    masks = {g: [] for g in GAMES}
    probs = {g: [] for g in GAMES}
    tgt_idx0 = {}
    for g in GAMES:
        for seed in SEEDS:
            enc = load_encoder(find_ckpt(CKPT_DIR / f"{g}_freeze-L0_seed{seed}"))
            e = enc.hrvq.quantizers[0].embedding.detach()
            assert torch.equal(e, emb0), f"pinned table differs: {g} seed {seed}"
            idx = level0_indices(enc, frames[g])
            if seed == 0:
                tgt_idx0[g] = idx
            p = usage(idx)
            a = p > TAU
            masks[g].append(a)
            probs[g].append(p)
            adopted = a & src_active
            inter = int(adopted.sum())
            z, m = hyper_z(inter, s_src, int(a.sum()), K)
            per_seed_rows.append({
                "game": g, "seed": seed,
                "active": int(a.sum()), "adopted": inter,
                "chance": round(m, 2), "z": round(float(z), 2),
                "mass_on_src": round(float(p[src_active].sum()), 3),
                "adopted_src_rank_med": round(float(src_rank[adopted].median()), 3),
                "norm_adopted": round(float(norms[adopted].mean()), 3),
                "norm_src_unadopted": round(float(norms[src_active & ~adopted].mean()), 3),
                "norm_inactive": round(float(norms[~src_active].mean()), 3),
            })

    maj = {g: (torch.stack(masks[g]).float().sum(0) >= 2) & src_active for g in GAMES}
    pair_rows = []
    for a, b in itertools.combinations(GAMES, 2):
        inter = int((maj[a] & maj[b]).sum())
        z, m = hyper_z(inter, int(maj[a].sum()), int(maj[b].sum()), s_src)
        pair_rows.append({"pair": f"{a}|{b}", "nA": int(maj[a].sum()),
                          "nB": int(maj[b].sum()), "inter": inter,
                          "chance": round(m, 2), "z": round(float(z), 2)})
    triple = maj[GAMES[0]] & maj[GAMES[1]] & maj[GAMES[2]]
    t_chance = float(maj[GAMES[0]].sum() * maj[GAMES[1]].sum() * maj[GAMES[2]].sum()) / (s_src * s_src)
    pair_rows.append({"pair": "triple", "nA": "", "nB": "",
                      "inter": int(triple.sum()), "chance": round(t_chance, 2), "z": ""})

    core = torch.where(triple)[0].tolist()
    code_rows = [{"code": c,
                  "pong_usage": round(float(p_src[c]), 5),
                  "pong_rank": round(float(src_rank[c]), 3),
                  "norm": round(float(norms[c]), 3),
                  **{f"{g}_mass_s{s}": round(float(probs[g][s][c]), 4)
                     for g in GAMES for s in SEEDS}}
                 for c in core]

    for name, rows in [("core_vocab_per_seed.csv", per_seed_rows),
                       ("core_vocab_pairs.csv", pair_rows),
                       ("core_vocab_codes.csv", code_rows)]:
        with open(OUT_DIR / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT_DIR / name} ({len(rows)} rows)")

    core_mass = {g: [round(float(probs[g][s][triple].sum()), 3) for s in SEEDS] for g in GAMES}
    print(json.dumps({"src_active": s_src, "core": core,
                      "pong_mass_on_core": round(float(p_src[triple].sum()), 3),
                      "target_mass_on_core": core_mass}, indent=1))

    if args.figures:
        make_figures(core, frames, src_idx, tgt_idx0)


def make_figures(core, frames, src_idx, tgt_idx0):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_games = ["pong"] + GAMES
    idx = {"pong": src_idx, **tgt_idx0}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(core), len(all_games),
                             figsize=(len(all_games) * 2.4, len(core) * 0.5))
    for i, c in enumerate(core):
        for j, g in enumerate(all_games):
            ax = axes[i, j]
            locs = (idx[g] == c).nonzero()
            if len(locs):
                sel = locs[rng.choice(len(locs), size=min(N_PATCH, len(locs)), replace=False)]
                ps = []
                for n, p in sel.tolist():
                    r, col = divmod(p, 4)
                    img = frames[g][n][-3:]
                    ps.append(img[:, r * 16:(r + 1) * 16, col * 16:(col + 1) * 16]
                              .permute(1, 2, 0).numpy().astype(np.uint8))
                gap = np.full((16, 2, 3), 255, np.uint8)
                strip = ps[0]
                for q in ps[1:]:
                    strip = np.concatenate([strip, gap, q], axis=1)
                ax.imshow(strip)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(str(c), rotation=0, ha="right", va="center", fontsize=7)
            if i == 0:
                ax.set_title(g.replace("_", " "), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "core_vocab_patches.png", dpi=180)
    print(f"wrote {FIG_DIR / 'core_vocab_patches.png'}")

    fig2, axs = plt.subplots(1, len(all_games), figsize=(len(all_games) * 2.2, 2.6))
    for j, g in enumerate(all_games):
        m = torch.zeros_like(idx[g], dtype=torch.bool)
        for c in core:
            m |= idx[g] == c
        heat = m.float().mean(0).reshape(4, 4).numpy()
        axs[j].imshow(heat, vmin=0, vmax=1, cmap="viridis")
        axs[j].set_title(g.replace("_", " "), fontsize=9)
        axs[j].set_xticks([]); axs[j].set_yticks([])
        for (r, cc), v in np.ndenumerate(heat):
            axs[j].text(cc, r, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=7)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "core_vocab_spatial.png", dpi=180)
    print(f"wrote {FIG_DIR / 'core_vocab_spatial.png'}")


if __name__ == "__main__":
    main()
