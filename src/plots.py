"""Charts. Every function takes the plain tables in outputs/ and writes one PNG."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (checked for colour-blind separation)
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0})


def _save(fig, title, note, path):
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.012, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sc_by_lap(t, name, path):
    """Chart 1. t: sc_model.by_lap's table. Left: the chance that a neutralisation starts on each lap. Right: the
    chance that at least one has started by each lap. The race in blue with its band, the average dry race in grey."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"left": 0.06, "right": 0.93, "top": 0.84, "bottom": 0.17, "wspace": 0.2})
    for ax, col, lo, hi, avg in ((a, "p", "p_lo", "p_hi", "avg"), (b, "cum", "cum_lo", "cum_hi", "avg_cum")):
        ax.fill_between(t.lap, t[lo], t[hi], color=BLUE, alpha=0.14, linewidth=0)
        ax.plot(t.lap, t[avg], color=MUTED, label="Average dry race, 2018-2026")
        ax.plot(t.lap, t[col], color=BLUE, label=f"{name} (band: 10th to 90th percentile)")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Lap")
        ax.set_xlim(1, t.lap.max())
        ax.set_ylim(0, None)
    a.set_title("Chance that a safety car, VSC or red flag starts on the lap")
    later = t[t.lap > 2]
    for lap, dy in ((1, 0), (2, 0)):
        a.annotate(f"lap {lap}: {t.p[lap - 1]:.0%}", (lap, t.p[lap - 1]), xytext=(8, dy), textcoords="offset points", va="center", color=INK2)
    a.annotate(f"every later lap: {later.p.mean():.1%}\n(average race: {later.avg.mean():.1%})", (later.lap.iloc[len(later) // 2], later.p.mean()),
               xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", color=INK2)
    b.set_title("Chance that at least one has started by the lap")
    b.set_ylim(0, 1)
    for col, colour, dy in (("cum", BLUE, 0), ("avg_cum", MUTED, 0)):
        b.plot(t.lap.iloc[-1], t[col].iloc[-1], "o", ms=8, color=colour, mec=SURFACE, mew=2, clip_on=False, zorder=5)
        b.annotate(f"{t[col].iloc[-1]:.0%}", (t.lap.iloc[-1], t[col].iloc[-1]), xytext=(9, dy), textcoords="offset points",
                   va="center", color=INK, fontweight="bold", annotation_clip=False)
    a.legend(loc="upper right", fontsize=9)
    _save(fig, f"When a neutralisation is likely: {name} against the average race",
          "Model: pooled logistic by lap and street / permanent, shrunk circuit and season multipliers. Dry races only. "
          "Source: FastF1 timing data 2018-2026.", path)


def sc_calibration(v, eo, path, bins=4):
    """Chart 2, on the hold-out seasons. v: sc_model.validate's race table. eo: sc_model.expected_observed's table.
    Left: races grouped by the model's P(at least one incident), against the share that had one. Right: incidents
    expected against observed, by kind, per test season and per part of the race."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"left": 0.07, "right": 0.97, "top": 0.85, "bottom": 0.15, "wspace": 0.5, "width_ratios": [1, 1.15]})
    g = v.groupby(pd.qcut(v.p_any, bins), observed=True).agg(p=("p_any", "mean"), share=("any", "mean"), n=("any", "size"))
    se = np.sqrt(g.share * (1 - g.share) / g.n)
    a.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    a.annotate("perfect calibration", (0.56, 0.56), rotation=45, rotation_mode="anchor", xytext=(4, 5), textcoords="offset points", color=MUTED, fontsize=8.5)
    a.errorbar(g.p, g.share, yerr=se, fmt="none", ecolor=BLUE, elinewidth=1.5)
    a.plot(g.p, g.share, "o", ms=9, color=BLUE, mec=SURFACE, mew=2)
    a.set(xlim=(0.5, 1), ylim=(0.32, 1.02), yticks=np.arange(0.4, 1.01, 0.1),
          xlabel="Model's chance of at least one neutralisation, before the race", ylabel="Share of those races that had one")
    a.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    a.set_title(f"Race by race: {len(v)} dry races of 2024-2026 in {bins} groups of {len(v) // bins}")

    kinds = {"SC": "SC or red flag", "VSC": "VSC"}
    rows = eo.assign(label=eo.group + " · " + eo.kind.map(kinds)).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows)) + (rows.by == "season") * 0.8           # a gap between the two blocks
    b.hlines(y, rows[["expected", "observed"]].min(axis=1), rows[["expected", "observed"]].max(axis=1), color=AXIS, linewidth=1.5)
    b.plot(rows.expected, y, "o", ms=8, color=MUTED, mec=SURFACE, mew=2, clip_on=False, label="Expected by the model")
    b.plot(rows.observed, y, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, clip_on=False, zorder=4, label="Happened")
    for yy, e, o in zip(y, rows.expected, rows.observed):
        b.annotate(f"{o:.0f} vs {e:.1f}", (max(e, o), yy), xytext=(9, 0), textcoords="offset points", va="center", color=INK2, fontsize=8.5)
    b.set_yticks(y, rows.label)
    b.set(xlim=(-1.2, rows[["expected", "observed"]].max().max() * 1.22), xlabel="Incidents on the hold-out laps")
    b.grid(axis="y", visible=False)
    b.set_title("Incidents: happened vs expected")
    b.legend(loc="upper right", fontsize=9)
    _save(fig, "Does the safety-car model hold up on seasons it has not seen?",
          "Each season predicted from the seasons before it; the season's own record builds up race by race. Bars on the left: one standard error. "
          "Source: FastF1 timing data 2018-2026.", path)
