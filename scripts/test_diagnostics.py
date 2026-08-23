"""Dependency-free smoke tests for the diagnostics pipeline.

Exercises the pure-Python logic of run_diagnostics.py (checkpoint resolution,
defaults merge, game filtering) and diagnostics_aggregate.py (per-seed
collapse, Jaccard-vs-source extraction, rho_enc, LaTeX/CSV emission) on
synthetic inputs - no torch, no GPU, no real checkpoints.

Run:  python scripts/test_diagnostics.py    # exits non-zero on first failure
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnostics_aggregate as agg  # numpy only - always importable

_CHECKS = 0


def check(cond, msg):
    global _CHECKS
    _CHECKS += 1
    if not cond:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# ----------------------------------------------------------------------------
# Synthetic results.json matching the schema written by mech_analysis_basic
# ----------------------------------------------------------------------------

def _row(label, game, cond, seed, active, resid, rho=None, drift_cos=None):
    usage = [{"active": a, "usage": a / 512, "perplexity": float(a)} for a in active]
    return {
        "label": label,
        "meta": {"dirname": label, "game": game, "condition": cond, "seed": seed},
        "usage": usage,
        "residual_errors": resid,
        "recon": [{"levels_used": list(range(l + 1)), "mse": 0.01 / (l + 1)} for l in range(3)],
        "drift": ([{"mean_l2": 0.1, "mean_rel_l2": 0.01,
                    "mean_matched_cos": drift_cos, "identity_rate": 1.0}] * 3
                  if drift_cos is not None else None),
        "enc_drift": ({"rho_enc_mean": rho, "rho_enc_median": rho} if rho is not None else None),
    }


def _pairwise(src, tgt, jaccs):
    return {"a": src, "b": tgt,
            "levels": [{"level": l, "jaccard": jaccs[l], "used_a": 10,
                        "used_b": 10, "mean_nn_cos": 0.5, "median_nn_cos": 0.5}
                       for l in range(3)]}


def make_results():
    rows = [
        _row("pong/scratch/s5", "pong", "scratch", 5, [60, 60, 60], [0.5, 0.5, 0.5]),
        _row("breakout/freeze-all/s5", "breakout", "freeze-all", 5,
             [40, 45, 50], [0.9, 0.8, 0.7], rho=1.0, drift_cos=0.95),
        _row("breakout/freeze-all/s6", "breakout", "freeze-all", 6,
             [42, 47, 52], [0.7, 0.6, 0.5], rho=0.5, drift_cos=0.85),
    ]
    pairwise = [
        _pairwise("pong/scratch/s5", "breakout/freeze-all/s5", [0.30, 0.20, 0.10]),
        _pairwise("pong/scratch/s5", "breakout/freeze-all/s6", [0.40, 0.30, 0.20]),
    ]
    return {"tag": "breakout", "rows": rows, "pairwise": pairwise}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_aggregate():
    print("[test] diagnostics_aggregate")
    data = make_results()
    groups = agg.collect_rows([("breakout", data)], source_game="pong")
    summary = agg.summarise(groups)

    key = ("breakout", "freeze-all")
    check(key in summary, "breakout/freeze-all present")
    check(("pong", "scratch") not in summary, "source row excluded from conditions")
    stats = summary[key]
    check(stats["n_seeds"] == 2, "n_seeds == 2")

    # active L0 mean = (40+42)/2 = 41
    check(abs(stats["active_l0"][0] - 41.0) < 1e-9, "active_l0 mean == 41")
    # Jaccard-vs-source L0 mean = (0.30+0.40)/2 = 0.35
    check(abs(stats["jaccard_src_l0"][0] - 0.35) < 1e-9, "jaccard_src_l0 mean == 0.35")
    # rho_enc mean = (1.0+0.5)/2 = 0.75
    check(abs(stats["rho_enc"][0] - 0.75) < 1e-9, "rho_enc mean == 0.75")
    # residual_err L0 mean = (0.9+0.7)/2 = 0.8
    check(abs(stats["residual_err_l0"][0] - 0.8) < 1e-9, "residual_err_l0 mean == 0.8")

    with tempfile.TemporaryDirectory() as d:
        agg.write_csv(summary, os.path.join(d, "a.csv"))
        agg.write_json(summary, os.path.join(d, "a.json"))
        agg.write_latex(summary, os.path.join(d, "t.tex"), "pong")
        tex = open(os.path.join(d, "t.tex")).read()
        check("freeze-all" in tex, "LaTeX names the condition")
        check("0.350" in tex, "LaTeX carries Jaccard value")
        check("0.750" in tex, "LaTeX carries rho_enc value")
        # The diagnostics data row (the one with rho_enc=0.750) must be fully
        # populated - no \tbd cells when source + drift + enc_drift are present.
        diag_row = [l for l in tex.splitlines()
                    if l.startswith(r"\textsc") and "0.750" in l]
        check(bool(diag_row) and r"\tbd" not in diag_row[0],
              "diagnostics data row has no \\tbd")


def test_cell():
    print("[test] _cell formatting")
    check(agg._cell(0.35, 0.05, 2, prec=3) == "0.350$\\pm$0.050", "mean+/-std cell")
    check(agg._cell(float("nan"), 0.0, prec=2) == r"\tbd", "NaN -> \\tbd")
    check(agg._cell(41.0, float("nan"), prec=0) == "41", "std NaN -> mean only")


def test_runner_resolution():
    print("[test] run_diagnostics resolution")
    try:
        import run_diagnostics as rd
    except Exception as e:  # heavy import (pulls nnet/torch); skip if unavailable
        print(f"  skip: run_diagnostics import unavailable ({type(e).__name__})")
        return

    check(rd.canonical_game("atari100k-ms_pacman") == "ms_pacman", "canonical_game strips prefix")
    check(rd.canonical_game(None) is None, "canonical_game(None) is None")

    with tempfile.TemporaryDirectory() as d:
        for name in ("breakout_freeze-all_seed5", "breakout_scratch_seed0",
                     "mspacman_scratch_seed0"):
            os.makedirs(os.path.join(d, name))
            open(os.path.join(d, name, "best.ckpt"), "w").close()

        g = rd.resolve_targets([os.path.join(d, "breakout_*", "best.ckpt")],
                               "breakout", warn=lambda *_: None)
        check(len(g) == 2, "glob resolves 2 breakout ckpts")

        r = rd.resolve_targets([{"root": d}], "breakout", warn=lambda *_: None)
        check(len(r) == 2, "root form filters to job game (breakout)")
        check(all("breakout" in os.path.dirname(p) for p in r), "root form excludes mspacman")

        missing = rd.resolve_targets([{"path": os.path.join(d, "nope", "best.ckpt")}],
                                     None, warn=lambda *_: None)
        check(missing == [], "missing path skipped")

    manifest = {
        "output_root": "OUT", "source": "SRC", "include_source": True,
        "frames": {"policy": "random", "n_frames": 256},
        "jobs": [{"name": "j", "targets": [], "frames": {"n_frames": 8}}],
    }
    job = rd.build_jobs(manifest)[0]
    check(job["source"] == "SRC", "job inherits default source")
    check(job["frames"]["n_frames"] == 8, "job overrides n_frames")
    check(job["frames"]["policy"] == "random", "job keeps default policy")


def main():
    print("== diagnostics smoke tests ==")
    test_cell()
    test_aggregate()
    test_runner_resolution()
    print(f"\nALL PASSED ({_CHECKS} checks)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
