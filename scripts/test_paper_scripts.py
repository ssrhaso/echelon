"""Dependency-free smoke tests for the inference / qualitative paper scripts.

Exercises the pure-Python and pandas/matplotlib logic of emit_active_codes.py,
plot_codebook_usage.py, and gen_reconstructions.py on synthetic inputs - no real
checkpoints and no GPU. The torch-importing parts of gen_reconstructions are
probed only through parse_dirname (pure regex); the heavy import is skipped if
torch/nnet are unavailable, mirroring scripts/test_diagnostics.py.

Run:  python scripts/test_paper_scripts.py    # exits non-zero on first failure
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import emit_active_codes as eac
import plot_codebook_usage as pcu

_CHECKS = 0


def check(cond, msg):
    global _CHECKS
    _CHECKS += 1
    if not cond:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def synthetic_agg():
    """One game x two conditions, with the active{l}_mean/std columns the
    emitter reads."""
    rows = []
    for cond, a, s in [("scratch", [320, 300, 310], [10, 8, 6]),
                       ("freeze-all", [170, 110, 100], [0, 5, 4])]:
        row = {"game": "demon_attack", "condition": cond, "n": 3}
        for l in range(3):
            row[f"active{l}_mean"] = a[l]
            row[f"active{l}_std"] = s[l]
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_raw():
    """Per-seed raw rows for the plot script (J{l} and active{l} columns)."""
    rows = []
    for game in pcu.GAMES:
        for cond in ("scratch", "freeze-all"):
            for seed in range(3):
                base = 0.2 if cond == "scratch" else 0.05
                row = {"game": game, "condition": cond, "seed": seed}
                for l in range(3):
                    row[f"J{l}"] = base - 0.01 * l + 0.001 * seed
                    row[f"active{l}"] = (300 if cond == "scratch" else 150) - 10 * l
                rows.append(row)
    return pd.DataFrame(rows)


def test_active_codes_cell():
    print("[test] emit_active_codes._cell")
    check(eac.cell(170.0, 14.0) == "170$\\pm$14", "mean+/-std rounds to whole codes")
    check(eac.cell(320.0, 0.0) == "320", "zero std -> mean only")
    check(eac.cell(float("nan"), 1.0) == r"\tbd", "NaN mean -> \\tbd")


def test_active_codes_build():
    print("[test] emit_active_codes.build")
    tex = eac.build(synthetic_agg())
    check(r"\label{tab:active_codes}" in tex, "table label present")
    check("170$\\pm$" in tex or "170" in tex, "freeze-all L0 count carried")
    check("from-scratch (ref)" in tex, "scratch row labelled")
    check(r"\tbd" not in tex, "no \\tbd when data present")


def test_plot_runs():
    print("[test] plot_codebook_usage.make_figure")
    df = synthetic_raw()
    with tempfile.TemporaryDirectory() as d:
        orig = pcu.OUT_DIR
        pcu.OUT_DIR = __import__("pathlib").Path(d)
        try:
            pcu.make_figure(df, "J", "J", "jacc")
            pcu.make_figure(df, "active", "A", "act")
        finally:
            pcu.OUT_DIR = orig
        files = os.listdir(d)
    check("jacc.png" in files and "jacc.pdf" in files, "jaccard figure written (png+pdf)")
    check("act.png" in files and "act.pdf" in files, "active-code figure written (png+pdf)")


def test_recon_dirname():
    print("[test] gen_reconstructions.parse_dirname")
    try:
        import gen_reconstructions as gr
    except Exception as e:  # heavy import (torch/nnet); skip if unavailable
        print(f"  skip: gen_reconstructions import unavailable ({type(e).__name__})")
        return
    check(gr.parse_dirname("demon_attack_freeze-all_seed2") == ("demon_attack", "freeze-all", 2),
          "parses game/condition/seed")
    check(gr.parse_dirname("up_n_down_scratch_seed0") == ("up_n_down", "scratch", 0),
          "handles multi-underscore game name")
    check(gr.parse_dirname("pong_source") is None, "non-matching dir -> None")


def main():
    print("== paper-script smoke tests ==")
    test_active_codes_cell()
    test_active_codes_build()
    test_plot_runs()
    test_recon_dirname()
    print(f"\nALL PASSED ({_CHECKS} checks)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
