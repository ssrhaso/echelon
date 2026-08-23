"""Manifest-driven mechanistic diagnostics over many checkpoints.

Wraps scripts/mech_analysis_basic.run_analysis() so that an entire batch of
checkpoints - across conditions, seeds, and target games - is declared once in
a small YAML/JSON manifest and run with a single command. Adding a checkpoint
is a one-line edit (or nothing at all, if a glob already covers it).

A manifest declares one or more *jobs*. Each job collects a shared held-out
frame batch from one env and scores every checkpoint in the job on those exact
frames (sha1-verified identical), which is what makes the per-level Jaccard,
quant-error, and codebook-drift numbers comparable across conditions.

Usage:
    python scripts/run_diagnostics.py --manifest configs/diagnostics/transfer.yaml
    python scripts/run_diagnostics.py --manifest m.yaml --only breakout   # one job
    python scripts/run_diagnostics.py --manifest m.yaml --dry_run         # resolve only

Manifest schema (top-level keys are per-job defaults; jobs may override):

    output_root: results/diagnostics      # per-job output -> <output_root>/<job>/
    source: path/to/pong_source/best.ckpt # reference ckpt for drift + Jaccard
    include_source: true                  # also score source on target frames
    plot: false                           # write summary + reconstruction PNGs
    frames:
      policy: random                      # random | agent
      n_frames: 256
      seed: 0

    jobs:
 - name: breakout
        frame_env: atari100k-breakout     # optional; inferred from 1st target
        targets:                          # any mix of the forms below
 - "checkpoints_local/breakout_*/best.ckpt"   # bare string == glob
 - glob: "checkpoints_local/breakout_*/best.ckpt"
 - path: "checkpoints_local/breakout_freeze-all_seed5/best.ckpt"
 - root: "checkpoints_local"     # discover */best.ckpt, filter to job game
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mech_analysis_basic as mech  # noqa: E402  (also puts repo root on path)


# ----------------------------------------------------------------------------
# Manifest loading + checkpoint resolution
# ----------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def canonical_game(frame_env):
    """'atari100k-ms_pacman' -> 'ms_pacman'; None -> None."""
    if not frame_env:
        return None
    return frame_env.split("atari100k-", 1)[-1]


def resolve_targets(entries, job_game, *, warn=print) -> list:
    """Resolve a list of target entries into an ordered, de-duplicated,
    existence-checked list of checkpoint paths.

    job_game (canonical, e.g. 'breakout') filters `root` auto-discovery so a
    shared checkpoint directory yields only the current job's game.
    """
    collected = []
    for e in entries or []:
        if isinstance(e, str):
            collected += sorted(glob.glob(e))
        elif isinstance(e, dict) and "path" in e:
            collected.append(e["path"])
        elif isinstance(e, dict) and "glob" in e:
            collected += sorted(glob.glob(e["glob"]))
        elif isinstance(e, dict) and "root" in e:
            for c in sorted(glob.glob(os.path.join(e["root"], "*", "best.ckpt"))):
                meta = mech.parse_ckpt_dirname(c)
                canon = mech.GAME_FROM_DIRNAME.get((meta["game"] or "").lower())
                if job_game is None or canon == job_game:
                    collected.append(c)
        else:
            raise ValueError(f"unrecognised target entry: {e!r}")

    seen, resolved = set(), []
    for p in collected:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if not os.path.exists(p):
            warn(f"  [warn] target not found, skipping: {p}")
            continue
        resolved.append(p)
    return resolved


def build_jobs(manifest: dict) -> list:
    """Merge per-job settings over manifest-level defaults into concrete jobs."""
    defaults = {
        "output_root": manifest.get("output_root", "results/diagnostics"),
        "source": manifest.get("source"),
        "include_source": manifest.get("include_source", True),
        "plot": manifest.get("plot", False),
        "frames": manifest.get("frames", {}) or {},
    }
    jobs = []
    for raw in manifest.get("jobs", []):
        frames = {**defaults["frames"], **(raw.get("frames") or {})}
        jobs.append({
            "name": raw["name"],
            "frame_env": raw.get("frame_env"),
            "targets": raw.get("targets", []),
            "source": raw.get("source", defaults["source"]),
            "include_source": raw.get("include_source", defaults["include_source"]),
            "plot": raw.get("plot", defaults["plot"]),
            "output_root": raw.get("output_root", defaults["output_root"]),
            "frames": frames,
        })
    return jobs


# ----------------------------------------------------------------------------
# Job execution
# ----------------------------------------------------------------------------

def run_job(job: dict, *, dry_run: bool = False) -> dict:
    name = job["name"]
    frame_env = job["frame_env"]
    job_game = canonical_game(frame_env)
    print(f"\n{'='*70}\n# JOB: {name}\n{'='*70}")

    targets = resolve_targets(job["targets"], job_game)
    if not targets:
        print(f"  [skip] no checkpoints resolved for job '{name}'")
        return {"name": name, "n_targets": 0, "ckpts": [], "output_dir": None}

    # Optionally score the source checkpoint on the target frames too, so the
    # source row appears in the pairwise Jaccard matrix (diagnostic D1).
    source = job["source"]
    ckpts = list(targets)
    if source and job["include_source"]:
        if os.path.exists(source):
            ckpts = [source] + [c for c in ckpts if os.path.abspath(c) != os.path.abspath(source)]
        else:
            print(f"  [warn] source not found, dropping from frame set: {source}")
    if source and not os.path.exists(source):
        source = None  # don't pass a missing source into drift computation

    output_dir = os.path.join(job["output_root"], name)
    frames = job["frames"]

    print(f"  frame_env       : {frame_env or '(infer from first ckpt)'}")
    print(f"  source ckpt     : {source or '(none)'}")
    print(f"  scored ckpts    : {len(ckpts)}")
    for c in ckpts:
        print(f" - {c}")
    print(f"  output_dir      : {output_dir}")
    print(f"  frames          : {frames or '(defaults)'}  plot={job['plot']}")

    if dry_run:
        return {"name": name, "n_targets": len(targets), "ckpts": ckpts,
                "output_dir": output_dir, "source": source}

    result = mech.run_analysis(
        ckpts=ckpts,
        source_ckpt=source,
        frame_env=frame_env,
        frame_policy=frames.get("policy", "random"),
        frame_policy_ckpt=frames.get("policy_ckpt"),
        n_frames=frames.get("n_frames", 256),
        seed=frames.get("seed", 0),
        frame_seed=frames.get("frame_seed"),
        output_dir=output_dir,
        tag=name,
        plot=job["plot"],
    )
    return {"name": name, "n_targets": len(targets), "ckpts": ckpts,
            "output_dir": output_dir, "source": source,
            "frame_sha1": result["frame_sha1"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Path to a YAML/JSON manifest.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these job name(s) from the manifest.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Resolve and print checkpoints per job without running.")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    jobs = build_jobs(manifest)
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j["name"] in wanted]
        missing = wanted - {j["name"] for j in jobs}
        if missing:
            ap.error(f"--only names not in manifest: {sorted(missing)}")
    if not jobs:
        ap.error("manifest declares no jobs")

    summary = [run_job(j, dry_run=args.dry_run) for j in jobs]

    print(f"\n{'='*70}\n# DONE: {len(summary)} job(s)\n{'='*70}")
    for s in summary:
        loc = s["output_dir"] or "(none)"
        print(f"  {s['name']:<16} {s['n_targets']:>3} ckpt(s) -> {loc}")


if __name__ == "__main__":
    main()
