"""Emit paper-ready LaTeX diagnostics tables from results/diagnostics_agg.csv.

Writes two files:

  results/diagnostics_table.tex      main text (Table: tab:results_diagnostics)
      Reports the two *scale-free* diagnostics, because neither raw metric can be
      compared across conditions on its own:
        J^(l)/J_0^(l)  Jaccard relative to a size-matched null. Raw J^(l) falls
                       with |A_tgt| under pure chance, and |A_tgt| spans ~50-350
                       across the ladder, so raw J confounds reuse with set size.
                       1.0 = chance, >1 = reuse, <1 = the target avoids the
                       source's cells. z is the hypergeometric z-score.
        Erel^(l)       quantisation error normalised by pre-VQ feature energy.
                       Raw E^(l) is an MSE in each checkpoint's own feature
                       space, so encoders differing only in feature scale give
                       E^(l) differing by its square.

  results/diagnostics_table_raw.tex  appendix (Table: tab:results_diagnostics_raw)
      The unnormalised J^(l), |A^(l)| and E^(l) behind the main table.

Usage:
  python scripts/emit_diag_latex.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
agg = pd.read_csv(ROOT / "results" / "diagnostics_agg.csv")
ORDER = ["scratch", "warmstart", "freeze-L0", "freeze-L01", "freeze-L012", "freeze-enc", "freeze-all"]
NAME = {"scratch": "from-scratch (ref)", "warmstart": r"\textsc{warmstart}",
        "freeze-L0": r"\textsc{freeze-L0}", "freeze-L01": r"\textsc{freeze-L01}",
        "freeze-L012": r"\textsc{freeze-L012}", "freeze-enc": r"\textsc{freeze-enc}",
        "freeze-all": r"\textsc{freeze-all}"}
GAME_TITLE = {"demon_attack": "Demon Attack", "ms_pacman": "Ms Pac-Man", "up_n_down": "Up'n'Down"}
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]

REQUIRED = [f"{m}{l}_mean" for m in ("Jratio", "Jz", "Erel") for l in range(3)]
missing = [c for c in REQUIRED if c not in agg.columns]
if missing:
    raise SystemExit(
        f"diagnostics_agg.csv lacks {missing[:3]}...\n"
        f"Run: python scripts/compute_diagnostics.py && python scripts/aggregate_diagnostics.py"
    )


def efmt(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


def block_rows(game):
    g = agg[agg.game == game].set_index("condition")
    return [(NAME[c], g.loc[c]) for c in ORDER if c in g.index]


def emit(path, label, caption, header, colspec, rowfn):
    L = [r"\begin{table}[t]", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
         r"    \centering\small\setlength{\tabcolsep}{4.5pt}",
         f"    \\begin{{tabular}}{{{colspec}}}", r"        \toprule"]
    L += header
    L.append(r"        \midrule")
    for gi, game in enumerate(GAMES):
        rows = block_rows(game)
        for i, (nm, r) in enumerate(rows):
            lead = (rf"\multirow{{{len(rows)}}}{{*}}{{\rotatebox{{90}}{{{GAME_TITLE[game]}}}}}"
                    if i == 0 else "")
            L.append(f"        {lead} & {nm:24s} & {rowfn(r)} \\\\")
        if gi < len(GAMES) - 1:
            L.append(r"        \midrule")
    L += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    out = ROOT / "results" / path
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}\n" + "\n".join(L) + "\n")


# ---------------------------------------------------------------- main table
emit(
    "diagnostics_table.tex", "tab:results_diagnostics",
    r"Per-level transfer diagnostics on a fixed held-out batch of target-game frames "
    r"($N{=}5000$ under a random policy, seed 42; the \emph{same} batch is encoded through "
    r"every condition's checkpoint). Each cell is the mean over 3 seeds. Both diagnostics are "
    r"reported \emph{scale-free}, since neither raw metric is comparable across conditions "
    r"(App.~\ref{app:per_level}, Table~\ref{tab:results_diagnostics_raw}). "
    r"$J^{(\ell)}/J^{(\ell)}_{0}$: code-set Jaccard divided by a size-matched null "
    r"($1.0=$ chance overlap for that active-set size, $>1$ $=$ reuse of source cells, "
    r"$<1$ $=$ the target lands in cells the source avoided); $z$ is the hypergeometric "
    r"z-score of the intersection ($|z|{>}2$ shaded in the text). "
    r"$\mathcal{E}^{(\ell)}_{\mathrm{rel}}$: target-frame quantization error as a fraction of "
    r"pre-VQ feature energy (lower $=$ the codebook absorbs more of the signal; "
    r"$1.0$ $=$ the codebook is inert). $\rho_{\text{enc}}$: cosine similarity between the "
    r"Pong-source and target encoder pre-VQ features.",
    [r"        & \multirow{2}{*}{Condition} & \multicolumn{3}{c}{Reuse $J^{(\ell)}/J^{(\ell)}_{0}$} "
     r"& \multicolumn{3}{c}{$z$} & \multicolumn{3}{c}{Residual $\mathcal{E}^{(\ell)}_{\mathrm{rel}}$ ($\downarrow$)} & Drift \\",
     r"        \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-12}",
     r"        & & L0 & L1 & L2 & L0 & L1 & L2 & L0 & L1 & L2 & $\rho_{\text{enc}}$ \\"],
    "ll ccc ccc ccc c",
    lambda r: " & ".join(
        [f"{r[f'Jratio{l}_mean']:.2f}" for l in range(3)]
        + [f"{r[f'Jz{l}_mean']:+.1f}" for l in range(3)]
        + [f"{r[f'Erel{l}_mean']:.2f}" for l in range(3)]
        + [f"{r['rho_enc_mean']:.2f}"]),
)

# ------------------------------------------------------------ appendix table
emit(
    "diagnostics_table_raw.tex", "tab:results_diagnostics_raw",
    r"Unnormalised diagnostics behind Table~\ref{tab:results_diagnostics} (mean over 3 seeds). "
    r"$J^{(\ell)}$: raw code-set Jaccard with the Pong source; $|A^{(\ell)}|$: target active-code "
    r"count (the source sets are $|A_{\text{Pong}}|=138/122/110$ at L0/L1/L2, of $K{=}512$); "
    r"$\mathcal{E}^{(\ell)}$: raw target-frame quantization error. Raw $J^{(\ell)}$ tracks "
    r"$|A^{(\ell)}|$ closely --- both fall roughly threefold from \textsc{freeze-enc} to "
    r"\textsc{freeze-all} --- which is why the main table divides it out; raw "
    r"$\mathcal{E}^{(\ell)}$ is an MSE in each checkpoint's own pre-VQ feature space and so "
    r"is not comparable across conditions with different encoders.",
    [r"        & \multirow{2}{*}{Condition} & \multicolumn{3}{c}{Jaccard $J^{(\ell)}$} "
     r"& \multicolumn{3}{c}{Active $|A^{(\ell)}|$} & \multicolumn{3}{c}{Quant.\ Error $\mathcal{E}^{(\ell)}$} \\",
     r"        \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
     r"        & & L0 & L1 & L2 & L0 & L1 & L2 & L0 & L1 & L2 \\"],
    "ll ccc ccc ccc",
    lambda r: " & ".join(
        [f"{r[f'J{l}_mean']:.3f}" for l in range(3)]
        + [f"{r[f'active{l}_mean']:.0f}" for l in range(3)]
        + [efmt(r[f"E{l}_mean"]) for l in range(3)]),
)
