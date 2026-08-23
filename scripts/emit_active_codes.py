"""Emit the per-level active-code count |A^(l)| table from the diagnostics.

The active-code count is the number of codes used above the threshold
tau = 1/(2K) on the fixed held-out target batch (the same active set used for
the Jaccard diagnostic). It is computed per seed in compute_diagnostics.py and
aggregated in aggregate_diagnostics.py; this script only formats it for the
paper, so no checkpoints or torch are required.

|A^(l)| is the missing companion to the per-level perplexity already reported in
the main table: perplexity is the *effective* code count, |A^(l)| the *literal*
count of codes above threshold. Reading them together separates "few codes used
heavily" from "many codes used evenly".

Reads:  results/diagnostics_agg.csv   (active{l}_mean / active{l}_std)
Writes: results/active_codes_table.tex

Usage:
  python scripts/emit_active_codes.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "diagnostics_agg.csv"
OUT = ROOT / "results" / "active_codes_table.tex"

ORDER = ["scratch", "warmstart", "freeze-L0", "freeze-L01", "freeze-L012", "freeze-enc", "freeze-all"]
NAME = {"scratch": "from-scratch (ref)", "warmstart": r"\textsc{warmstart}",
        "freeze-L0": r"\textsc{freeze-L0}", "freeze-L01": r"\textsc{freeze-L01}",
        "freeze-L012": r"\textsc{freeze-L012}", "freeze-enc": r"\textsc{freeze-enc}",
        "freeze-all": r"\textsc{freeze-all}"}
GAME_TITLE = {"demon_attack": "Demon Attack", "ms_pacman": "Ms Pac-Man", "up_n_down": "Up'n'Down"}
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]


def cell(mean, std):
    """mean +/- std rounded to whole codes; std omitted when degenerate."""
    if pd.isna(mean):
        return r"\tbd"
    if pd.isna(std) or std == 0:
        return f"{mean:.0f}"
    return f"{mean:.0f}$\\pm${std:.0f}"


def build(agg):
    L = [r"\begin{table}[t]",
         r"    \caption{Per-level active-code count $|A^{(\ell)}|$ on the fixed held-out "
         r"target batch ($N{=}5000$, random policy, seed 42), mean $\pm$ std over 3 seeds. "
         r"A code is active when its usage exceeds $\tau=1/(2K)=1/1024$. $K=512$ bounds each "
         r"cell from above. Read alongside the perplexity column of "
         r"Table~\ref{tab:results_diagnostics}: a low $|A^{(\ell)}|$ under the fully-frozen "
         r"conditions (\textsc{freeze-L012}, \textsc{freeze-all}) confirms the target collapses "
         r"onto a narrow subset of the frozen Pong cells.}",
         r"    \label{tab:active_codes}",
         r"    \centering\small\setlength{\tabcolsep}{6pt}",
         r"    \begin{tabular}{ll ccc}",
         r"        \toprule",
         r"        & Condition & $|A^{(0)}|$ & $|A^{(1)}|$ & $|A^{(2)}|$ \\",
         r"        \midrule"]
    for gi, game in enumerate(GAMES):
        g = agg[agg.game == game].set_index("condition")
        block = []
        for c in ORDER:
            if c not in g.index:
                continue
            r = g.loc[c]
            cells = " & ".join(cell(r[f"active{l}_mean"], r[f"active{l}_std"]) for l in range(3))
            block.append((NAME[c], cells))
        nrow = len(block)
        for i, (nm, cells) in enumerate(block):
            lead = (rf"\multirow{{{nrow}}}{{*}}{{\rotatebox{{90}}{{{GAME_TITLE[game]}}}}}"
                    if i == 0 else "")
            L.append(f"        {lead} & {nm:24s} & {cells} \\\\")
        if gi < len(GAMES) - 1:
            L.append(r"        \midrule")
    L += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    if not AGG.exists():
        raise SystemExit(f"missing {AGG} - run aggregate_diagnostics.py first")
    agg = pd.read_csv(AGG)
    tex = build(agg)
    OUT.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT}\n")
    print(tex)


if __name__ == "__main__":
    main()
