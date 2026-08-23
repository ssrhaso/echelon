#!/usr/bin/env python
"""Appendix table: ECHELON from-scratch on Atari 100k against published baselines.

Every finished seed is used, with no selection. Sources:

  * wandb_export/eval_epoch_long.csv        -- the six study games (full curves)
  * results/scratch_baseline_wandb_status.csv -- near_final_return, finished seeds

Statistic: near-final return = mean of a seed's last five evaluation points, the
same statistic used everywhere else in the paper.

Four of the 26 games (Crazy Climber, Qbert, Road Runner, Seaquest) are omitted:
only a final-evaluation-point value survives locally for them, not a curve, so
the near-final statistic cannot be computed. Aggregates are therefore computed
over the 22 reported games for every method, which is why they differ from the
26-game aggregates the baseline papers publish.

Published per-game scores verified against:
  IRIS       Micheli et al., ICLR 2023, Table 1        (arXiv:2209.00588)
  Delta-IRIS Micheli et al., ICML 2024, Table 8        (arXiv:2406.19320)
  TWISTER    Burchi & Timofte, ICLR 2025, Table 2      (arXiv:2503.04416)
Random / Human are the IRIS Table 1 one-decimal reference column, which is the
unrounded source the other two papers round.
"""
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "tex")
os.makedirs(TEX, exist_ok=True)

# game -> (display, random, human, IRIS, Delta-IRIS, TWISTER)
BASE = {
    "alien":           ("Alien",            227.8,  7127.7,   420.0,   391,   970),
    "amidar":          ("Amidar",             5.8,  1719.5,   143.0,    64,   184),
    "assault":         ("Assault",          222.4,   742.0,  1524.4,  1123,   721),
    "asterix":         ("Asterix",          210.0,  8503.3,   853.6,  2492,  1306),
    "bank_heist":      ("Bank Heist",        14.2,   753.1,    53.1,  1148,   942),
    "battle_zone":     ("Battle Zone",     2360.0, 37187.5, 13074.0, 11825,  9920),
    "boxing":          ("Boxing",             0.1,    12.1,    70.1,    70,    88),
    "breakout":        ("Breakout",           1.7,    30.5,    83.7,   302,    35),
    "chopper_command": ("Chopper Command",  811.0,  7387.8,  1565.0,  1183,   910),
    "demon_attack":    ("Demon Attack",     152.1,  1971.0,  2034.4,   533,   289),
    "freeway":         ("Freeway",            0.0,    29.6,    31.1,    31,    32),
    "frostbite":       ("Frostbite",         65.2,  4334.7,   259.1,   279,   305),
    "gopher":          ("Gopher",           257.6,  2412.5,  2236.1,  6445, 22234),
    "hero":            ("Hero",            1027.0, 30826.4,  7037.4,  7049,  8773),
    "jamesbond":       ("James Bond",        29.0,   302.8,   462.7,   309,   573),
    "kangaroo":        ("Kangaroo",          52.0,  3035.0,   838.2,  2269,  6016),
    "krull":           ("Krull",           1598.0,  2665.5,  6616.4,  5978,  8839),
    "kung_fu_master":  ("Kung Fu Master",   258.5, 22736.3, 21759.8, 21534, 23442),
    "ms_pacman":       ("Ms Pac-Man",       307.3,  6951.6,   999.1,  1067,  2206),
    "pong":            ("Pong",             -20.7,    14.6,    14.6,    20,    20),
    "private_eye":     ("Private Eye",       24.9, 69571.3,   100.0,   103,  1608),
    "up_n_down":       ("Up'n'Down",        533.4, 11693.2,  3546.2,  4072,  7068),
}


def echelon_scores():
    out = {}
    with open(os.path.join(ROOT, "analysis_pta", "returns_per_seed.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["condition"] == "from-scratch" and r["complete"] == "1"
                    and r["evidence"] == "wandb_export" and r["nearfinal"]):
                out.setdefault(r["game"], []).append(float(r["nearfinal"]))
    with open(os.path.join(ROOT, "results", "scratch_baseline_wandb_status.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["state"] == "finished" and r.get("near_final_return"):
                out.setdefault(r["game"], []).append(float(r["near_final_return"]))
    return out


def fmt(v):
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return "{:,.0f}".format(v).replace(",", "{,}")
    if a >= 100:
        return "{:.0f}".format(v)
    return "{:.1f}".format(v)


def main():
    ours = echelon_scores()
    games = [g for g in BASE if g in ours]
    missing = sorted(set(BASE) - set(games))
    assert not missing, "no ECHELON scores for " + str(missing)

    rows, hns = [], {"IRIS": [], "dIRIS": [], "TWISTER": [], "ECHELON": []}
    sup = {k: 0 for k in hns}
    for g in sorted(games, key=lambda k: BASE[k][0]):
        name, rnd, hum, iris, diris, tw = BASE[g]
        v = ours[g]
        mean = st.mean(v)
        vals = {"IRIS": iris, "dIRIS": float(diris), "TWISTER": float(tw), "ECHELON": mean}
        for k, x in vals.items():
            n = (x - rnd) / (hum - rnd)
            hns[k].append(n)
            sup[k] += int(n >= 1.0)
        best = max(vals.values())
        second = sorted(set(vals.values()), reverse=True)[1]

        def cell(k):
            x = vals[k]
            s = fmt(x)
            if x == best:
                return r"\textbf{" + s + "}"
            if x == second:
                return r"\textit{" + s + "}"
            return s

        rows.append("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            name, fmt(rnd), fmt(hum), cell("IRIS"), cell("dIRIS"), cell("TWISTER"),
            cell("ECHELON"), len(v)))

    def agg(fn, key):
        vals = {k: fn(hns[k]) for k in hns}
        best = max(vals.values())
        second = sorted(set(vals.values()), reverse=True)[1]
        out = []
        for k in ("IRIS", "dIRIS", "TWISTER", "ECHELON"):
            s = "{:.3f}".format(vals[k]) if key != "sup" else str(vals[k])
            if vals[k] == best:
                s = r"\textbf{" + s + "}"
            elif vals[k] == second:
                s = r"\textit{" + s + "}"
            out.append(s)
        return out

    rows.append(r"\midrule")
    supvals = {k: sup[k] for k in sup}
    best, second = max(supvals.values()), sorted(set(supvals.values()), reverse=True)[1]
    scells = []
    for k in ("IRIS", "dIRIS", "TWISTER", "ECHELON"):
        s = str(supvals[k])
        if supvals[k] == best:
            s = r"\textbf{" + s + "}"
        elif supvals[k] == second:
            s = r"\textit{" + s + "}"
        scells.append(s)
    rows.append("Superhuman games ($\\uparrow$) & 0 & n/a & {} & n/a \\\\".format(
        " & ".join(scells)))
    rows.append("Mean HNS ($\\uparrow$) & 0.000 & 1.000 & {} & n/a \\\\".format(
        " & ".join(agg(st.mean, "mean"))))
    rows.append("Median HNS ($\\uparrow$) & 0.000 & 1.000 & {} & n/a \\\\".format(
        " & ".join(agg(st.median, "med"))))

    head = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rr rrr r c}",
        r"\toprule",
        r"Game & Random & Human & IRIS & $\Delta$-IRIS & TWISTER & ECHELON & $n$ \\",
        r"\midrule",
    ]
    tail = [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TEX, "tab_suite.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + rows + tail) + "\n")
    print("wrote tex/tab_suite.tex  ({} games)".format(len(games)))
    for k in ("IRIS", "dIRIS", "TWISTER", "ECHELON"):
        print("  {:<9} superhuman={:>2}  mean HNS={:.3f}  median HNS={:.3f}".format(
            k, sup[k], st.mean(hns[k]), st.median(hns[k])))
    print("  seeds per ECHELON cell:",
          ", ".join("{}={}".format(BASE[g][0], len(ours[g])) for g in sorted(games)))


if __name__ == "__main__":
    main()
