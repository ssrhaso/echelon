#!/usr/bin/env python
"""Read-only view of results/diagnostics_agg.csv under the corrected condition names.

Emits analysis_pta/diagnostics_view.csv with exactly the quantities the paper
reports, so every diagnostic number in the TeX traces to one row here.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis_pta")

RENAME = {"scratch": "from-scratch", "warmstart": "adapt-CB012", "freeze-L0": "freeze-CB0",
          "freeze-L01": "freeze-CB01", "freeze-L012": "freeze-CB012",
          "freeze-enc": "adapt-CB012+encinit", "freeze-all": "freeze-CB012+encinit"}
ORDER = ["from-scratch", "adapt-CB012", "freeze-CB0", "freeze-CB01", "freeze-CB012",
         "adapt-CB012+encinit", "freeze-CB012+encinit"]
GAMES = ["demon_attack", "ms_pacman", "up_n_down"]

FIELDS = ["Jratio0", "Jratio1", "Jratio2", "Jz0", "Jz1", "Jz2",
          "Erel0", "Erel1", "Erel2", "zenergy", "rho_enc",
          "active0", "active1", "active2", "J0", "J1", "J2",
          "E0", "E1", "E2", "perp0", "perp1", "perp2"]


def main():
    src = {}
    with open(os.path.join(ROOT, "results", "diagnostics_agg.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            src[(r["game"], RENAME[r["condition"]])] = r

    rows = []
    for g in GAMES:
        for c in ORDER:
            r = src[(g, c)]
            out = {"game": g, "condition": c, "n": r["n"]}
            for f in FIELDS:
                v = r.get(f + "_mean")
                out[f] = "" if v in (None, "") else round(float(v), 4)
                s = r.get(f + "_std")
                out[f + "_sd"] = "" if s in (None, "") else round(float(s), 4)
            rows.append(out)

    with open(os.path.join(OUT, "diagnostics_view.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = "{:<14}{:<22}{:>6}{:>6}{:>6}{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}{:>9}{:>7}"
    print(hdr.format("game", "condition", "Jr0", "Jr1", "Jr2", "z0", "z1", "z2",
                     "Er0", "Er1", "Er2", "R0", "rho"))
    for r in rows:
        print("{:<14}{:<22}{:>6.2f}{:>6.2f}{:>6.2f}{:>+7.1f}{:>+7.1f}{:>+7.1f}"
              "{:>7.2f}{:>7.2f}{:>7.2f}{:>9.0f}{:>7.2f}".format(
                  r["game"], r["condition"], r["Jratio0"], r["Jratio1"], r["Jratio2"],
                  r["Jz0"], r["Jz1"], r["Jz2"], r["Erel0"], r["Erel1"], r["Erel2"],
                  r["zenergy"], r["rho_enc"]))
    act = [r[f] for r in rows for f in ("active0", "active1", "active2")]
    print("\nactive-set size range over all cells: {:.0f} to {:.0f}".format(min(act), max(act)))
    print("wrote analysis_pta/diagnostics_view.csv")


if __name__ == "__main__":
    main()
