"""README summary figure from the recorded sanity run (sanity.cast + nvidia-smi csv).

Usage: python make_hero.py [--cast sanity.cast] [--gpu-csv gpu_mem.csv] [--out hero.png]
"""
import argparse, csv, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, INK, INK2, GRID, SURF, TILE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff", "#f3f3f0"

ap = argparse.ArgumentParser()
ap.add_argument("--cast", default="sanity.cast"); ap.add_argument("--gpu-csv", default="gpu_mem.csv")
ap.add_argument("--out", default="hero.png")
args = ap.parse_args()

plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9})

txt = "".join(json.loads(l)[2] for l in open(args.cast) if l.startswith("["))
times = np.array([float(t) for _, t in re.findall(r"(FLARE23Ts_\d+): ([\d.]+) s", txt)])
rows = list(csv.reader(open(args.gpu_csv)))[1:]
t = np.array([float(r[0]) for r in rows]); mem = np.array([float(r[1]) for r in rows])
# per-case peak GPU memory: walk the 1 Hz log with the per-case durations, starting at the first rise
cur = t[next(i for i, v in enumerate(mem) if v > 100)]
peaks = []
for s in times:
    seg = mem[(t >= cur) & (t < cur + s)]
    peaks.append(seg.max() if len(seg) else np.nan); cur += s
peaks = np.array(peaks)

fig = plt.figure(figsize=(12.5, 6.0), facecolor=SURF)
fig.text(0.04, 0.945, "PANTHER  ·  FLARE 2026 Task 1", fontsize=15, fontweight="bold", color=INK)
fig.text(0.04, 0.905, "5-fold plain U-Net ensemble with a Mahalanobis mask-shape gate: accurate, abstains on healthy scans, fits the inference limits",
         fontsize=10, color=INK2)

# KPI tiles
kpis = [("75.38 %", "lesion DSC", "hidden validation"), ("70.09 %", "lesion NSD", "hidden validation"),
        ("8.1 %", "false-positive rate", "172 healthy CTs (was 100 %)"), ("9.2 s", "median time per scan", "RTX 3090, max 29.3 s"),
        ("2614 MB", "peak GPU memory", "tolerance 4096 MB")]
x0, w, gap, y, h = 0.04, 0.176, 0.008, 0.66, 0.19
for i, (v, l1, l2) in enumerate(kpis):
    x = x0 + i * (w + gap)
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                      facecolor=TILE, edgecolor="none"))
    fig.text(x + 0.014, y + 0.105, v, fontsize=21, fontweight="bold", color=INK)
    fig.text(x + 0.014, y + 0.062, l1, fontsize=9.5, color=INK)
    fig.text(x + 0.014, y + 0.025, l2, fontsize=8.5, color=INK2)

gs = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.55, bottom=0.12, wspace=0.38)

def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

ax = fig.add_subplot(gs[0, 0]); style(ax)
labels, vals = ["single fold", "5-fold\nensemble", "ensemble\n+ shape gate"], [100.0, 69.8, 8.1]
bars = ax.bar(labels, vals, width=0.58, color=BLUE)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:g} %", ha="center", fontsize=9.5, color=INK)
ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("healthy CTs with a non-empty mask, %")
ax.set_title("False positives on 172 healthy CT scans", loc="left")

ax = fig.add_subplot(gs[0, 1]); style(ax)
o = np.sort(times); ax.bar(np.arange(len(o)), o, width=0.75, color=BLUE)
ax.axhline(60, color=INK2, ls="--", lw=1); ax.text(0, 61.5, "60 s cutoff", fontsize=8.5, color=INK2)
ax.axhline(45, color=INK2, ls=":", lw=1); ax.text(0, 46.5, "45 s tolerance", fontsize=8.5, color=INK2)
ax.text(len(o) - 1, o.max() + 2, f"max {o.max():.1f} s", ha="right", fontsize=9, color=INK)
ax.set_ylim(0, 70); ax.set_xticks([]); ax.set_xlabel("50 public validation scans, sorted"); ax.set_ylabel("running time per scan, s")
ax.set_title(f"Inference time per scan (median {np.median(times):.1f} s)", loc="left")

ax = fig.add_subplot(gs[0, 2]); style(ax)
o = np.sort(peaks); ax.bar(np.arange(len(o)), o, width=0.75, color=BLUE)
ax.axhline(4096, color=INK2, ls="--", lw=1); ax.text(0, 4180, "4096 MB tolerance", fontsize=8.5, color=INK2)
ax.text(len(o) - 1, np.nanmax(o) + 90, f"max {np.nanmax(o):.0f} MB", ha="right", fontsize=9, color=INK)
ax.set_ylim(0, 4700); ax.set_xticks([]); ax.set_xlabel("50 public validation scans, sorted"); ax.set_ylabel("peak GPU memory per scan, MB")
ax.set_title("GPU memory per scan (nvidia-smi, 1 Hz)", loc="left")

fig.savefig(args.out, dpi=200, facecolor=SURF); print("saved", args.out)
