#!/usr/bin/env python3
"""Compare ablation (no-validation) vs standard runs for paper figure."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "paper" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── load knowledge histories ──────────────────────────────────────────
def load_history(path):
    with open(path) as f:
        d = json.load(f)
    confs = defaultdict(list)
    for c in d.get("clusters", []):
        confs[c["cluster_type"]].append(c["confidence"])
    for r in d.get("relations", []):
        confs[r["relation_type"]].append(r["confidence"])
    for p in d.get("paths", []):
        confs[p.get("path_type", "path")].append(p["confidence"])
    all_c = [v for vs in confs.values() for v in vs]
    return confs, all_c


orig_by_type, orig_all = load_history(ROOT / "output" / ".knowledge_history.json")
abl_by_type, abl_all = load_history(ROOT / "output_ablation_noval" / ".knowledge_history.json")

# ── metadata ──────────────────────────────────────────────────────────
with open(ROOT / "output" / "xch_run_20260220_163743" / "metadata.json") as f:
    orig_meta = json.load(f)
with open(ROOT / "output_ablation_noval" / "xch_run_20260409_121937" / "metadata.json") as f:
    abl_meta = json.load(f)


# ── print comparison table ────────────────────────────────────────────
def tier(c):
    if c >= 0.70:
        return "Validated"
    if c >= 0.60:
        return "Pending"
    return "Rejected"


def tier_pcts(confs):
    n = len(confs)
    if n == 0:
        return 0, 0, 0
    v = sum(1 for c in confs if c >= 0.70)
    p = sum(1 for c in confs if 0.60 <= c < 0.70)
    r = sum(1 for c in confs if c < 0.60)
    return v / n * 100, p / n * 100, r / n * 100


print("=" * 70)
print(f"{'Metric':<35} {'Standard':>15} {'No-Valid.':>15}")
print("=" * 70)
print(f"{'Model':<35} {'gemini-2.5-flash':>15}")
print(
    f"{'Total assertions (run 1)':<35} {orig_meta['assertions']['total']:>15} {abl_meta['assertions']['total']:>15}"
)
print(
    f"{'Total triples':<35} {orig_meta['triples']['total']:>15} {abl_meta['triples']['total']:>15}"
)
print(
    f"{'Duration (s)':<35} {orig_meta['timing']['duration_seconds']:>15.1f} {abl_meta['timing']['duration_seconds']:>15.1f}"
)
print("-" * 70)
print(f"{'Knowledge items (cumul.)':<35} {len(orig_all):>15} {len(abl_all):>15}")
print(f"{'Mean confidence':<35} {np.mean(orig_all):>15.3f} {np.mean(abl_all):>15.3f}")
print(f"{'Std confidence':<35} {np.std(orig_all):>15.3f} {np.std(abl_all):>15.3f}")
print(f"{'Min confidence':<35} {min(orig_all):>15.3f} {min(abl_all):>15.3f}")
print(f"{'Max confidence':<35} {max(orig_all):>15.3f} {max(abl_all):>15.3f}")
ov, op, orr = tier_pcts(orig_all)
av, ap, ar = tier_pcts(abl_all)
print(f"{'Validated (≥0.70)':<35} {ov:>14.1f}% {av:>14.1f}%")
print(f"{'PendingReview (0.60-0.69)':<35} {op:>14.1f}% {ap:>14.1f}%")
print(f"{'Rejected (<0.60)':<35} {orr:>14.1f}% {ar:>14.1f}%")
print("=" * 70)

# ── Figure: side-by-side confidence distributions ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

bins = np.arange(0.45, 1.05, 0.05)

axes[0].hist(orig_all, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].set_title("Standard (α=0.7, β=0.3)", fontsize=11)
axes[0].set_xlabel("Confidence score")
axes[0].set_ylabel("Count")
axes[0].axvline(0.70, color="green", ls="--", lw=1.2, label="Validated ≥0.70")
axes[0].axvline(0.60, color="orange", ls="--", lw=1.2, label="Pending ≥0.60")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0.45, 1.02)

axes[1].hist(abl_all, bins=bins, color="#FF5722", edgecolor="white", alpha=0.85)
axes[1].set_title("Ablation (α=1.0, β=0.0)", fontsize=11)
axes[1].set_xlabel("Confidence score")
axes[1].axvline(0.70, color="green", ls="--", lw=1.2, label="Validated ≥0.70")
axes[1].axvline(0.60, color="orange", ls="--", lw=1.2, label="Pending ≥0.60")
axes[1].legend(fontsize=8)
axes[1].set_xlim(0.45, 1.02)

fig.suptitle("Ablation Study: Effect of Validation Weighting on Confidence", fontsize=13, y=1.02)
plt.tight_layout()
out_path = METRICS_DIR / "ablation_confidence.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")

# ── Figure 2: stacked bar by type ────────────────────────────────────
all_types = sorted(set(list(orig_by_type.keys()) + list(abl_by_type.keys())))
fig2, ax2 = plt.subplots(figsize=(9, 4))
x = np.arange(len(all_types))
w = 0.35
orig_means = [np.mean(orig_by_type.get(t, [0])) for t in all_types]
abl_means = [np.mean(abl_by_type.get(t, [0])) for t in all_types]

bars1 = ax2.bar(x - w / 2, orig_means, w, label="Standard", color="#2196F3", alpha=0.85)
bars2 = ax2.bar(x + w / 2, abl_means, w, label="Ablation (no valid.)", color="#FF5722", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(all_types, rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Mean Confidence")
ax2.set_title("Mean Confidence by Knowledge Type", fontsize=12)
ax2.axhline(0.70, color="green", ls="--", lw=1, alpha=0.6)
ax2.legend()
ax2.set_ylim(0.4, 1.05)
plt.tight_layout()
out_path2 = METRICS_DIR / "ablation_by_type.png"
fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"Figure saved: {out_path2}")
