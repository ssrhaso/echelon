"""Emit the isca-aug26 run manifest: one row per (game, condition, seed).

The manifest is the single source of truth for the SLURM array. Row N of the CSV
(0-indexed, excluding the header) is array task N, so `--array=0-139` covers the whole
sprint and `--array=0-83` covers just the seed-0..2 block.

Ordering (matters - a truncated window must still leave complete ladders):
  1. seed block: seeds 0-2 for ALL games before any seed 3-4, so stopping early
     yields a complete n=3 ladder on every game rather than n=5 on some and n=0
     on others.
  2. game priority: ms_pacman > asterix > up_n_down > demon_attack
     (ms_pacman kills the host-stack confound, asterix is the missing H/L cell,
     up_n_down needs n=5, demon_attack is a redo of an already-valid ladder).
  3. condition, then seed.

Precision is a single value across the entire manifest by construction. Mixing
precision (or host) within a game's cells is what invalidated the Ms Pac-Man ladder;
`--precision` applies to every row and the launcher re-checks it.

Usage:
    python scripts/gen_isca_manifest.py --precision fp32
    python scripts/gen_isca_manifest.py --precision bf16 --out experiments/isca_aug26_manifest.csv
"""
import argparse
import csv
import os

# Priority order (see the ordering note above). Index = launch priority.
GAMES = ["ms_pacman", "asterix", "up_n_down", "demon_attack"]

# Games valid for --games beyond the aug26 four. Breakout is here for the
# weight-transfer baseline, whose only rich codebook-transfer ladder (n=9-10)
# is the original Pong->Breakout one; it is NOT in GAMES so a default aug26
# regen cannot silently gain a fifth game.
KNOWN_GAMES = GAMES + ["breakout"]

SEEDS = [0, 1, 2, 3, 4]

# The 7 aug26 cells: from-scratch + the 6 transfer conditions. Flag mapping is taken
# from experiments/transfer_freezing.yaml + setup_transfer.ps1 (Get-LaunchArgs).
#   name -> (freeze_levels, freeze_encoder, needs_transfer_checkpoint, transfer_all)
#
# 'weight-transfer' is the XTRA-style baseline added after the aug26 sprint: it
# warm-starts every task-agnostic weight rather than just the codebooks, which is
# the comparison a reader needs to judge whether codebook transfer is doing
# anything a plain weight transfer would not. It is NOT in DEFAULT_CONDITIONS, so
# regenerating the aug26 manifest does not silently gain an eighth cell -- pass
# `--conditions weight-transfer` to emit it on its own.
CONDITIONS = [
    ("scratch",         "",      False, False, False),
    ("warmstart",       "",      False, True,  False),
    ("freeze-L0",       "0",     False, True,  False),
    ("freeze-L01",      "0,1",   False, True,  False),
    ("freeze-L012",     "0,1,2", False, True,  False),
    ("freeze-enc",      "",      True,  True,  False),
    ("freeze-all",      "0,1,2", True,  True,  False),
    ("weight-transfer", "",      False, True,  True),
    # Control: freeze all three codebooks at their RANDOM initialisation (no
    # donor checkpoint at all). Distinguishes "returns tolerate ANY inert
    # codebook" from "returns tolerate a Pong-trained one" -- the disambiguation
    # tab:results_main cannot currently make. Not in DEFAULT_CONDITIONS.
    ("freeze-random",   "0,1,2", False, False, False),
]

DEFAULT_CONDITIONS = [c[0] for c in CONDITIONS
                      if c[0] not in ("weight-transfer", "freeze-random")]

# The protected transfer source. Read-only.
# Staged to this path on ISCA by experiments/isca_setup.sh.
PONG_ARTIFACT = "haso-university-of-the-west-of-england/nnet/pong-source-checkpoints:pong-seed5-source"
PONG_CKPT = "transfer_ckpt/pong_seed5_best.ckpt"

# override_config JSON per precision. fp32 is the codebase default (twister.py sets
# atari100k -> float32) but we write it explicitly anyway so the value is never
# implicit: an empty override is indistinguishable from "nobody chose", which is what
# made the precision of the earlier runs unrecoverable without grepping stdout.
PRECISION_OVERRIDE = {
    "fp32": '{"precision":"float32"}',
    "bf16": '{"precision":"bfloat16"}',
}

FIELDS = [
    "idx", "game", "condition", "seed", "precision", "tag", "wandb_name",
    "env_name", "run_name", "freeze_levels", "freeze_encoder",
    "transfer_checkpoint", "transfer_all", "override_config", "seed_block",
    "game_priority",
]


def wandb_name(game, condition, seed, source="pong"):
    """Run name. The pong default keeps the established prefixes so the
    analysis pulls classify these runs without modification; other sources and
    the freeze-random control get their own prefixes."""
    if condition == "scratch":
        return "baseline/{}/seed{}".format(game, seed)
    if condition == "freeze-random":
        return "control/freeze-random/{}/seed{}".format(game, seed)
    return "transfer/{}to/{}/{}/seed{}".format(source, game, condition, seed)


def build_rows(precision, tag, games=GAMES, seeds=SEEDS, conditions=None,
               source_name="pong", source_ckpt=PONG_CKPT):
    override = PRECISION_OVERRIDE[precision]
    wanted = set(conditions) if conditions is not None else set(DEFAULT_CONDITIONS)
    rows = []
    for seed in seeds:
        for gi, game in enumerate(games):
            for cond, levels, freeze_enc, needs_ckpt, transfer_all in CONDITIONS:
                if cond not in wanted:
                    continue
                rows.append({
                    "game": game,
                    "condition": cond,
                    "seed": seed,
                    "precision": precision,
                    "tag": tag,
                    "wandb_name": wandb_name(game, cond, seed, source_name),
                    "env_name": "atari100k-{}".format(game),
                    "run_name": "hrvq_baseline" if cond == "scratch" else "hrvq_transfer",
                    "freeze_levels": levels,
                    "freeze_encoder": "1" if freeze_enc else "0",
                    "transfer_checkpoint": source_ckpt if needs_ckpt else "",
                    "transfer_all": "1" if transfer_all else "0",
                    "override_config": override,
                    "seed_block": 0 if seed <= 2 else 1,
                    "game_priority": gi,
                })
    # seeds 0-2 for every game before any seed 3-4; then game priority, condition, seed.
    cond_index = {c[0]: i for i, c in enumerate(CONDITIONS)}
    rows.sort(key=lambda r: (r["seed_block"], r["game_priority"],
                             cond_index[r["condition"]], r["seed"]))
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=sorted(PRECISION_OVERRIDE), required=True,
                    help="ONE precision for every isca-aug26 run. Never mix.")
    ap.add_argument("--tag", default="isca-aug26", help="W&B tag applied to every run")
    ap.add_argument("--out", default="experiments/isca_aug26_manifest.csv")
    ap.add_argument("--games", nargs="*", default=GAMES,
                    help="subset of games, in launch-priority order")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS,
                    help="subset of cells to emit (default: the 7 aug26 cells; pass "
                         "'weight-transfer' for the weight-transfer baseline alone, "
                         "'freeze-random' for the frozen-random-codebook control)")
    ap.add_argument("--source_name", default="pong",
                    help="donor tag used in run names: transfer/<source>to/... "
                         "(default pong)")
    ap.add_argument("--source_ckpt", default=PONG_CKPT,
                    help="staged donor checkpoint path on ISCA for the transfer "
                         "conditions (default {})".format(PONG_CKPT))
    args = ap.parse_args()

    for g in args.games:
        if g not in KNOWN_GAMES:
            ap.error("unknown game {!r}; expected a subset of {}".format(g, KNOWN_GAMES))

    known_conditions = [c[0] for c in CONDITIONS]
    for c in args.conditions:
        if c not in known_conditions:
            ap.error("unknown condition {!r}; expected a subset of {}".format(c, known_conditions))

    rows = build_rows(args.precision, args.tag, args.games, args.seeds, args.conditions,
                      args.source_name, args.source_ckpt)

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})

    # Summary: the block boundaries are what you pass to --array.
    n_block0 = sum(1 for r in rows if r["seed_block"] == 0)
    print("wrote {} ({} rows, precision={}, tag={})".format(
        args.out, len(rows), args.precision, args.tag))
    print("  seed 0-2 block : array indices 0-{}  ({} runs)".format(n_block0 - 1, n_block0))
    if n_block0 < len(rows):
        print("  seed 3-4 block : array indices {}-{}  ({} runs)".format(
            n_block0, len(rows) - 1, len(rows) - n_block0))
    print()
    print("  per game (priority order):")
    for gi, g in enumerate(args.games):
        idxs = [r["idx"] for r in rows if r["game"] == g]
        lo3 = [r["idx"] for r in rows if r["game"] == g and r["seed_block"] == 0]
        print("    P{} {:<14} {:>3} runs   seeds0-2: {}-{}   all: {}-{}".format(
            gi + 1, g, len(idxs), min(lo3), max(lo3), min(idxs), max(idxs)))
    print()
    if any(r["transfer_checkpoint"] for r in rows):
        if args.source_ckpt == PONG_CKPT:
            print("  transfer source (read-only): {}".format(PONG_ARTIFACT))
        print("  staged at                  : {}".format(args.source_ckpt))
    else:
        print("  no donor checkpoint needed (scratch / freeze-random rows only)")


if __name__ == "__main__":
    main()
