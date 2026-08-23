#!/usr/bin/env python
r"""Emit every numeric LaTeX table and figure body used by pta_submission.tex.

Every number in the paper's tables and in the decoupling figure is written by
this script from analysis_pta/*.csv, so nothing is hand-typed. Output lands in
tex/.

Each table is emitted as a COMPLETE tabular environment rather than as a row
fragment. A bare fragment cannot be \input inside a tabular: the \\ ending the
preceding row scans ahead for its optional argument, expands \input during that
scan, and the following \midrule or \bottomrule then lands outside the
alignment ("Misplaced \noalign").
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "analysis_pta")
TEX = os.path.join(ROOT, "tex")
os.makedirs(TEX, exist_ok=True)

NL = r"\\"

REF = {"breakout": (1.7, 30.5), "demon_attack": (152.1, 1971.0),
       "ms_pacman": (307.3, 6951.6), "up_n_down": (533.4, 11693.2)}
GAMENAME = {"breakout": "Breakout", "demon_attack": "Demon Attack",
            "ms_pacman": "Ms Pac-Man", "up_n_down": "Up'n'Down",
            "pong": "Pong", "freeway": "Freeway"}
LADDER = ["from-scratch", "adapt-CB012", "freeze-CB0", "freeze-CB01",
          "freeze-CB012", "adapt-CB012+encinit", "freeze-CB012+encinit"]
CONTROLS = ["freeze-CBrand012", "whole-model-transfer",
            "adapt-CB012 (DA donor)", "freeze-CB012 (DA donor)"]
CTRLNAME = {"freeze-CBrand012": r"\cond{freeze-CBrand012}",
            "whole-model-transfer": r"\cond{whole-model-transfer}",
            "adapt-CB012 (DA donor)": r"\cond{adapt-CB012} (D.A.)",
            "freeze-CB012 (DA donor)": r"\cond{freeze-CB012} (D.A.)"}


def cond(c):
    return r"\cond{" + c.replace("+", "{+}") + "}"


def rd(path):
    with open(os.path.join(A, path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v, g):
    """Game-appropriate return formatting; thousands separator safe in math mode."""
    v = float(v)
    if g == "breakout":
        return "{:.1f}".format(v)
    if abs(v) >= 1000:
        return "{:,.0f}".format(v).replace(",", "{,}")
    return "{:.0f}".format(v)


def rowlead(g, k, span):
    if k:
        return ""
    return r"\multirow{" + str(span) + r"}{*}{\rotatebox{90}{" + GAMENAME[g] + "}}"


def write(name, lines, colspec=None, header=None, pre=None):
    body = []
    if colspec is not None:
        if pre:
            body.extend(pre)
        body.append(r"\begin{tabular}{" + colspec + "}")
        body.append(r"\toprule")
        if header:
            body.extend(header)
        body.append(r"\midrule")
    body.extend(lines)
    if colspec is not None:
        body.append(r"\bottomrule")
        body.append(r"\end{tabular}")
    with open(os.path.join(TEX, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    print("wrote tex/" + name)


# --------------------------------------------------------- main: actionability
def table_actionability():
    iqm = {r["condition"]: r for r in rd("iqm_hns.csv")}
    poi = {r["condition"]: r for r in rd("prob_improvement.csv")}
    out = []
    for c in LADDER:
        if c not in iqm:
            continue
        i, p = iqm[c], poi.get(c)
        pcell = "n/a" if p is None else "${:.2f}$ [{:.2f}, {:.2f}]".format(
            float(p["poi_vs_scratch"]), float(p["ci_lo"]), float(p["ci_hi"]))
        out.append("{} & {} & ${:.2f}$ [{:.2f}, {:.2f}] & {} {}".format(
            cond(c), i["n_runs"], float(i["iqm_hns"]), float(i["ci_lo"]),
            float(i["ci_hi"]), pcell, NL))
        if c == "from-scratch":
            out.append(r"\midrule")
    write("tab_actionability.tex", out, colspec="lccc",
          header=[r"Condition & $n$ seeds & IQM of normalized return"
                  r" & $P(\text{improvement})$ " + NL])


# --------------------------------------------------------- main: diagnostics
def table_diagnostics():
    by = {(r["game"], r["condition"]): r for r in rd("diagnostics_view.csv")}
    out = []
    for gi, g in enumerate(["demon_attack", "ms_pacman", "up_n_down"]):
        if gi:
            out.append(r"\midrule")
        for k, c in enumerate(LADDER):
            r = by[(g, c)]
            out.append("{} & {} & {:.2f} & {:.2f} & {:.2f} & ${:+.1f}$ & ${:+.1f}$"
                       " & ${:+.1f}$ & {:.2f} & {:.2f} & {:.2f} & {} {}".format(
                           rowlead(g, k, 7), cond(c),
                           float(r["Jratio0"]), float(r["Jratio1"]), float(r["Jratio2"]),
                           float(r["Jz0"]), float(r["Jz1"]), float(r["Jz2"]),
                           float(r["Erel0"]), float(r["Erel1"]), float(r["Erel2"]),
                           num(r["zenergy"], g), NL))
    write("tab_diagnostics.tex", out, colspec="ll ccc ccc ccc c",
          pre=[r"\setlength{\tabcolsep}{4pt}"],
          header=[r"& \multirow{2}{*}{Condition}"
                  r" & \multicolumn{3}{c}{Reuse $J^{(\ell)}/J^{(\ell)}_{0}$}"
                  r" & \multicolumn{3}{c}{$z$}"
                  r" & \multicolumn{3}{c}{$\Erel^{(\ell)}$ ($\downarrow$)}"
                  r" & $R^{(0)}$ " + NL,
                  r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
                  r"\cmidrule(lr){9-11}\cmidrule(lr){12-12}",
                  r"& & L0 & L1 & L2 & L0 & L1 & L2 & L0 & L1 & L2 & " + NL])


# --------------------------------------------------------- main: controls
def table_controls():
    ctrl = {(r["game"], r["condition"]): r for r in rd("controls.csv")}
    games = ["demon_attack", "ms_pacman", "up_n_down"]
    out = []
    for c in CONTROLS:
        cells = []
        for g in games:
            r = ctrl.get((g, c))
            cells.append("n/a" if r is None else "{} $\\pm$ {}".format(
                num(r["mean"], g), num(r["std"], g)))
        out.append("{} & {} {}".format(CTRLNAME[c], " & ".join(cells), NL))
        if c == "whole-model-transfer":
            out.append(r"\addlinespace[2pt]")
            out.append(r"\multicolumn{4}{l}{\emph{Second donor: Demon Attack}} " + NL)
    write("tab_controls.tex", out, colspec="lccc",
          header=[r"Condition & Demon Attack & Ms Pac-Man & Up'n'Down " + NL])


# --------------------------------------------------------- main: figure body
def figure_decoupling():
    diag = {(r["game"], r["condition"]): r for r in rd("diagnostics_view.csv")}
    cells = {(r["game"], r["condition"]): r for r in rd("per_game_cells.csv")}
    style = {"demon_attack": ("*", "draw=echA, fill=echA!35"),
             "up_n_down": ("square*", "draw=echB, fill=echB!35"),
             "ms_pacman": ("triangle", "draw=echC, fill=echC!10")}
    out = []
    for g, (mark, sty) in style.items():
        pts = []
        for c in LADDER:
            d, s = diag.get((g, c)), cells.get((g, c))
            if not (d and s):
                continue
            r0, h0 = REF[g]
            pts.append("({:.3f},{:.3f})".format(
                float(d["Erel0"]), (float(s["median"]) - r0) / (h0 - r0)))
        out.append("% " + g)
        out.append(r"\addplot[only marks, mark=" + mark + ", mark size=2.1pt, "
                   + sty + "] coordinates {" + " ".join(pts) + "};")
        out.append(r"\addlegendentry{" + GAMENAME[g] + "}")
    write("fig_decoupling_data.tex", out)


# --------------------------------------------------------- appendix tables
def table_per_game_full():
    by = {(r["game"], r["condition"]): r for r in rd("per_game_cells.csv")}
    out = []
    for gi, g in enumerate(["breakout", "demon_attack", "ms_pacman", "up_n_down"]):
        if gi:
            out.append(r"\midrule")
        for k, c in enumerate(LADDER):
            r = by.get((g, c))
            if r is None:
                continue
            out.append("{} & {} & {} & {} [{}, {}] & {} $\\pm$ {} {}".format(
                rowlead(g, k, 7), cond(c), r["n"], num(r["median"], g),
                num(r["iqr_lo"], g), num(r["iqr_hi"], g), num(r["mean"], g),
                num(r["std"], g) if r["std"] else "n/a", NL))
    write("tab_per_game_full.tex", out, colspec="ll c c c",
          header=[r"& Condition & $n$ & Median [IQR] & Mean $\pm$ SD " + NL])


def table_diag_raw():
    by = {(r["game"], r["condition"]): r for r in rd("diagnostics_view.csv")}
    out = []
    for gi, g in enumerate(["demon_attack", "ms_pacman", "up_n_down"]):
        if gi:
            out.append(r"\midrule")
        for k, c in enumerate(LADDER):
            r = by[(g, c)]
            out.append("{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.0f} & {:.0f} & {:.0f}"
                       " & {} & {} & {} & {:.0f} & {:.0f} & {:.0f} & ${:+.2f}$ {}".format(
                           rowlead(g, k, 7), cond(c),
                           float(r["J0"]), float(r["J1"]), float(r["J2"]),
                           float(r["active0"]), float(r["active1"]), float(r["active2"]),
                           num(r["E0"], g), num(r["E1"], g), num(r["E2"], g),
                           float(r["perp0"]), float(r["perp1"]), float(r["perp2"]),
                           float(r["rho_enc"]), NL))
    write("tab_diag_raw.tex", out, colspec="ll ccc ccc ccc ccc c",
          pre=[r"\setlength{\tabcolsep}{2.0pt}"],
          header=[r"& \multirow{2}{*}{Condition} & \multicolumn{3}{c}{$J^{(\ell)}$}"
                  r" & \multicolumn{3}{c}{$|A^{(\ell)}|$}"
                  r" & \multicolumn{3}{c}{$\mathcal{E}^{(\ell)}$}"
                  r" & \multicolumn{3}{c}{$P^{(\ell)}$}"
                  r" & $\rho_{\mathrm{enc}}$ " + NL,
                  r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}"
                  r"\cmidrule(lr){12-14}\cmidrule(lr){15-15}",
                  r"& & L0 & L1 & L2 & L0 & L1 & L2 & L0 & L1 & L2"
                  r" & L0 & L1 & L2 & " + NL])


def table_per_seed():
    rows = [r for r in rd("returns_per_seed.csv") if r["complete"] == "1"]
    by = {}
    for r in rows:
        by.setdefault((r["game"], r["condition"]), []).append(float(r["nearfinal"]))
    out = []
    order = ["pong", "breakout", "demon_attack", "ms_pacman", "up_n_down", "freeway"]
    first = True
    for g in order:
        conds = [c for c in LADDER + CONTROLS if (g, c) in by]
        if not conds:
            continue
        if not first:
            out.append(r"\midrule")
        first = False
        for k, c in enumerate(conds):
            v = sorted(by[(g, c)])
            lead = (r"\multirow{" + str(len(conds)) + r"}{*}{" + GAMENAME[g] + "}") if k == 0 else ""
            out.append("{} & {} & {} & {} {}".format(
                lead, CTRLNAME.get(c, cond(c)), len(v),
                ", ".join(num(x, g) for x in v), NL))
    write("tab_per_seed.tex", out, colspec=r"ll c p{0.45\linewidth}",
          header=[r"Game & Condition & $n$ & Near-final return per seed " + NL])


def table_stack_audit():
    out = []
    for r in rd("stack_audit.csv"):
        mixed = r["mixed_cells"]
        mixed = "none" if mixed == "-" else mixed.replace(";", ", ").replace("+", "{+}")
        out.append("{} & {} & {} & {} {}".format(
            GAMENAME[r["game"]], r["n_stacks"], mixed,
            "single stack" if r["verdict"] == "single-stack" else "mixed, excluded", NL))
    write("tab_stack_audit.tex", out, colspec=r"l c p{0.36\linewidth} l",
          header=[r"Game & Configurations & Cells mixing configurations"
                  r" & Verdict " + NL])


if __name__ == "__main__":
    table_actionability()
    table_diagnostics()
    table_controls()
    figure_decoupling()
    table_per_game_full()
    table_diag_raw()
    table_per_seed()
    table_stack_audit()
