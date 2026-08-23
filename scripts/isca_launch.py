"""Validate the isca-aug26 manifest and emit (or submit) the sbatch array commands.

Dry-run is the default: nothing is submitted unless you pass --submit.

    # what would go up, whole sprint
    python scripts/isca_launch.py

    # the pilot gate: exactly 2 runs (Ms Pac-Man scratch + freeze-L012, seed 0)
    python scripts/isca_launch.py --pilot --submit

    # seed 0-2 block only, 32 concurrent
    python scripts/isca_launch.py --seed-block 0 --concurrent 32 --submit

    # one game
    python scripts/isca_launch.py --games ms_pacman --submit

Guards, all of which refuse to submit rather than warn:
  * every row shares ONE precision  -- mixing precision within a game's cells is what
    invalidated the Ms Pac-Man ladder;
  * every row shares ONE tag;
  * a game is submitted as a whole ladder, never a partial top-up across stacks;
  * the manifest's idx column matches row order (the SLURM array indexes into it).
"""
import argparse
import csv
import os
import subprocess
import sys

SLURM_SCRIPT = "experiments/isca_aug26.slurm"
DEFAULT_MANIFEST = "experiments/isca_aug26_manifest.csv"

# Ms Pac-Man scratch seed0 + Ms Pac-Man freeze-L012 seed0: the cheapest pair that
# exercises both ends of the ladder before the full array goes up.
PILOT_CELLS = [("ms_pacman", "scratch", 0), ("ms_pacman", "freeze-L012", 0)]


def compress(indices):
    """[0,1,2,5,6,9] -> '0-2,5-6,9' (SLURM array notation)."""
    idx = sorted(set(indices))
    if not idx:
        return ""
    parts, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else "{}-{}".format(start, prev))
        start = prev = i
    parts.append(str(start) if start == prev else "{}-{}".format(start, prev))
    return ",".join(parts)


def load(manifest):
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("ERROR: manifest {} is empty".format(manifest))
    for i, r in enumerate(rows):
        if int(r["idx"]) != i:
            sys.exit("ERROR: manifest idx column is out of order at row {} (idx={}). "
                     "The SLURM array indexes into this file by position; regenerate "
                     "with scripts/gen_isca_manifest.py.".format(i, r["idx"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--submit", action="store_true",
                    help="actually sbatch (default is dry-run: print only)")
    ap.add_argument("--account", default=None,
                    help="SLURM account; default reads it from experiments/isca_stitch.slurm")
    ap.add_argument("--games", nargs="*", default=None, help="restrict to these games")
    ap.add_argument("--seed-block", type=int, choices=[0, 1], default=None,
                    help="0 = seeds 0-2, 1 = seeds 3-4")
    ap.add_argument("--concurrent", type=int, default=32,
                    help="max simultaneous array tasks (%%N). 16 GPUs x 2 = 32 slots")
    ap.add_argument("--pilot", action="store_true",
                    help="submit only the 2 pilot runs and stop")
    ap.add_argument("--per-game", action="store_true",
                    help="emit one sbatch per game instead of a single array")
    args = ap.parse_args()

    rows = load(args.manifest)

    # ---- global invariants ----
    precisions = sorted({r["precision"] for r in rows})
    if len(precisions) != 1:
        sys.exit("ERROR: manifest mixes precisions {}. Every isca-aug26 run must share "
                 "one precision.".format(precisions))
    tags = sorted({r["tag"] for r in rows})
    if len(tags) != 1:
        sys.exit("ERROR: manifest mixes tags {}.".format(tags))
    precision, tag = precisions[0], tags[0]

    # ---- select rows ----
    sel = rows
    if args.pilot:
        want = set(PILOT_CELLS)
        sel = [r for r in rows if (r["game"], r["condition"], int(r["seed"])) in want]
        if len(sel) != len(PILOT_CELLS):
            sys.exit("ERROR: pilot cells not all present in manifest (found {} of {})".format(
                len(sel), len(PILOT_CELLS)))
    else:
        if args.games:
            sel = [r for r in sel if r["game"] in args.games]
        if args.seed_block is not None:
            sel = [r for r in sel if int(r["seed_block"]) == args.seed_block]
    if not sel:
        sys.exit("ERROR: selection is empty")

    # ---- partial-ladder guard ----
    # A game must go up as a complete ladder for the seeds selected. Topping up a
    # subset of cells is exactly the cross-stack mixing that broke Ms Pac-Man.
    if not args.pilot:
        for g in sorted({r["game"] for r in sel}):
            all_cells = {(r["condition"], r["seed"]) for r in rows if r["game"] == g
                         and (args.seed_block is None
                              or int(r["seed_block"]) == args.seed_block)}
            got = {(r["condition"], r["seed"]) for r in sel if r["game"] == g}
            missing = all_cells - got
            if missing:
                sys.exit("ERROR: selection would submit a PARTIAL ladder for {}: missing "
                         "{}. Submit the whole ladder or none of it.".format(
                             g, sorted(missing)[:6]))

    account = args.account
    if account is None:
        try:
            with open("experiments/isca_stitch.slurm", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#SBATCH -A"):
                        account = line.split()[2]
                        break
        except OSError:
            pass
    if not account or account == "REPLACE_WITH_ACCOUNT_CODE":
        account = "REPLACE_WITH_ACCOUNT_CODE"
        print("WARNING: no SLURM account resolved. Find it on ISCA with:")
        print("    sacctmgr -n show assoc user=$USER format=account%40")
        print("    ...then pass --account <code>.\n")

    # ---- build the sbatch command(s) ----
    groups = {}
    if args.per_game or args.pilot:
        for r in sel:
            groups.setdefault(r["game"], []).append(int(r["idx"]))
    else:
        groups["isca-aug26"] = [int(r["idx"]) for r in sel]

    cmds = []
    for label, idxs in groups.items():
        spec = compress(idxs)
        if args.concurrent and not args.pilot:
            spec += "%{}".format(args.concurrent)
        cmds.append([
            "sbatch",
            "-A", account,
            "--array={}".format(spec),
            "--export=ALL,MANIFEST={}".format(args.manifest),
            "-J", "aug26-{}".format(label),
            SLURM_SCRIPT,
        ])

    # ---- report ----
    print("=" * 78)
    print("isca-aug26 launch plan   [{}]".format("SUBMIT" if args.submit else "DRY RUN"))
    print("=" * 78)
    print("manifest  : {}".format(args.manifest))
    print("precision : {}   (single value across all rows - verified)".format(precision))
    print("tag       : {}".format(tag))
    print("account   : {}".format(account))
    print("selected  : {} runs".format(len(sel)))
    for g in sorted({r["game"] for r in sel}):
        grows = [r for r in sel if r["game"] == g]
        seeds = sorted({int(r["seed"]) for r in grows})
        conds = sorted({r["condition"] for r in grows})
        print("    {:<14} {:>3} runs   {} cells x seeds {}".format(
            g, len(grows), len(conds), seeds))
    ckpts = {r["transfer_checkpoint"] for r in sel if r["transfer_checkpoint"]}
    if ckpts:
        print("transfer source (read-only, must be staged on ISCA):")
        for c in sorted(ckpts):
            print("    {}".format(c))
    print()
    for c in cmds:
        print("  " + " ".join(c))
    print()

    if not args.submit:
        print("Dry run: nothing submitted. Re-run with --submit to launch.")
        return

    if account == "REPLACE_WITH_ACCOUNT_CODE":
        sys.exit("REFUSING to submit without a real SLURM account code (--account).")

    for c in cmds:
        print("submitting: " + " ".join(c), flush=True)
        res = subprocess.run(c, capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode != 0:
            sys.exit("sbatch failed with exit {}".format(res.returncode))


if __name__ == "__main__":
    main()
