"""
Compute Metrics — Quantitative analysis of xch-MIND runs for camera-ready paper.

Produces:
  1. Validation Pass Rate per agent (proposed vs novel vs filtered)
  2. Assertion distribution per tier (Validated / PendingReview / Rejected)
  3. Cross-run Jaccard stability analysis
  4. Saturation curve (assertions per run)
  5. Confidence calibration plot (c_LLM vs c_val) — requires instrumented runs
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Optional: matplotlib for plots
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib not found — tables only, no plots")


BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "paper" / "metrics"


def load_metadata_runs() -> list[dict]:
    """Load all metadata.json files sorted chronologically."""
    runs = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        meta = d / "metadata.json"
        if d.is_dir() and meta.exists():
            with open(meta) as f:
                data = json.load(f)
            data["_dir"] = d.name
            runs.append(data)
    return runs


def load_knowledge_history() -> dict:
    """Load the accumulated knowledge history."""
    path = OUTPUT_DIR / ".knowledge_history.json"
    if not path.exists():
        print("⚠ .knowledge_history.json not found")
        return {}
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Validation Pass Rate (Novelty Filtering)
# ─────────────────────────────────────────────────────────────────────────────


def compute_novelty_rates(runs: list[dict]) -> dict:
    """
    Per-agent aggregation: total proposed, total novel, total filtered.
    'Novel' means the LLM proposal passed both novelty detection AND
    deterministic validation to be committed.
    """
    agents = ["geo_analyzer", "temporal_analyzer", "type_analyzer", "path_generator"]
    totals = {a: {"proposed": 0, "novel": 0, "filtered": 0} for a in agents}

    for run in runs:
        nf = run.get("novelty_filtering", {})
        for agent in agents:
            stats = nf.get(agent, {})
            totals[agent]["proposed"] += stats.get("proposed", 0)
            totals[agent]["novel"] += stats.get("novel", 0)
            totals[agent]["filtered"] += stats.get("filtered_as_duplicates", 0)

    return totals


def print_novelty_table(totals: dict):
    agent_labels = {
        "geo_analyzer": "GeoAnalyzer",
        "temporal_analyzer": "TemporalAnalyzer",
        "type_analyzer": "TypeAnalyzer",
        "path_generator": "PathGenerator",
    }
    print("\n" + "=" * 72)
    print("METRIC 1: Validation Pass Rate (across all runs)")
    print("=" * 72)
    print(f"{'Agent':<20} {'Proposed':>10} {'Accepted':>10} {'Filtered':>10} {'Accept%':>10}")
    print("-" * 72)

    total_proposed = 0
    total_novel = 0
    total_filtered = 0

    for agent, label in agent_labels.items():
        p = totals[agent]["proposed"]
        n = totals[agent]["novel"]
        filt = totals[agent]["filtered"]
        rate = (n / p * 100) if p > 0 else 0
        print(f"{label:<20} {p:>10} {n:>10} {filt:>10} {rate:>9.1f}%")
        total_proposed += p
        total_novel += n
        total_filtered += filt

    rate = (total_novel / total_proposed * 100) if total_proposed > 0 else 0
    print("-" * 72)
    print(
        f"{'TOTAL':<20} {total_proposed:>10} {total_novel:>10} {total_filtered:>10} {rate:>9.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Confidence Tier Distribution
# ─────────────────────────────────────────────────────────────────────────────


def compute_confidence_tiers(history: dict) -> dict:
    """Classify all committed assertions by confidence tier."""
    tiers = {"Validated (≥0.70)": 0, "PendingReview (0.60–0.69)": 0, "Rejected (<0.60)": 0}
    by_type = defaultdict(lambda: {"Validated": 0, "PendingReview": 0, "Rejected": 0})

    all_items = []
    for cluster in history.get("clusters", []):
        all_items.append((cluster["cluster_type"], cluster["confidence"]))
    for relation in history.get("relations", []):
        all_items.append((relation["relation_type"], relation["confidence"]))
    for path in history.get("paths", []):
        all_items.append((path.get("path_type", "path"), path["confidence"]))

    for item_type, conf in all_items:
        if conf >= 0.70:
            tiers["Validated (≥0.70)"] += 1
            by_type[item_type]["Validated"] += 1
        elif conf >= 0.60:
            tiers["PendingReview (0.60–0.69)"] += 1
            by_type[item_type]["PendingReview"] += 1
        else:
            tiers["Rejected (<0.60)"] += 1
            by_type[item_type]["Rejected"] += 1

    return {"overall": tiers, "by_type": dict(by_type), "total": len(all_items)}


def print_tier_table(tiers_data: dict):
    print("\n" + "=" * 72)
    print("METRIC 2: Confidence Tier Distribution (committed assertions)")
    print("=" * 72)

    total = tiers_data["total"]
    print(f"\nOverall (n={total}):")
    for tier, count in tiers_data["overall"].items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {tier:<28} {count:>6}  ({pct:.1f}%)")

    print(f"\n{'Type':<22} {'Validated':>10} {'Pending':>10} {'Rejected':>10}")
    print("-" * 56)
    for item_type, counts in sorted(tiers_data["by_type"].items()):
        print(
            f"{item_type:<22} {counts['Validated']:>10} "
            f"{counts['PendingReview']:>10} {counts['Rejected']:>10}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Cross-run Jaccard Stability
# ─────────────────────────────────────────────────────────────────────────────


def compute_jaccard_stability(history: dict) -> dict:
    """
    Compute Jaccard similarity between consecutive runs.
    For clusters: compare sets of frozenset(member_uris).
    For relations: compare sets of (source, target, type).
    """
    # Group by run
    runs_clusters = defaultdict(list)
    runs_relations = defaultdict(list)
    runs_paths = defaultdict(list)

    for c in history.get("clusters", []):
        runs_clusters[c["run_id"]].append(frozenset(c["member_uris"]))
    for r in history.get("relations", []):
        runs_relations[r["run_id"]].append((r["source_uri"], r["target_uri"], r["relation_type"]))
    for p in history.get("paths", []):
        runs_paths[p["run_id"]].append(frozenset(p["stop_uris"]))

    run_ids = sorted(
        set(list(runs_clusters.keys()) + list(runs_relations.keys()) + list(runs_paths.keys()))
    )

    if len(run_ids) < 2:
        return {"message": "Need ≥2 runs for stability analysis"}

    # Cumulative Jaccard: how much does each new run ADD vs what's already known?
    results = []
    cumulative_clusters = set()
    cumulative_relations = set()

    for i, run_id in enumerate(run_ids):
        new_clusters = set(runs_clusters.get(run_id, []))
        new_relations = set(runs_relations.get(run_id, []))

        if i == 0:
            cumulative_clusters = new_clusters
            cumulative_relations = new_relations
            results.append(
                {
                    "run": run_id,
                    "new_clusters": len(new_clusters),
                    "new_relations": len(new_relations),
                    "cumulative_clusters": len(cumulative_clusters),
                    "cumulative_relations": len(cumulative_relations),
                    "cluster_novelty": 1.0,
                    "relation_novelty": 1.0,
                }
            )
        else:
            overlap_c = new_clusters & cumulative_clusters
            overlap_r = new_relations & cumulative_relations
            novelty_c = (len(new_clusters) - len(overlap_c)) / max(1, len(new_clusters))
            novelty_r = (len(new_relations) - len(overlap_r)) / max(1, len(new_relations))

            cumulative_clusters |= new_clusters
            cumulative_relations |= new_relations

            results.append(
                {
                    "run": run_id,
                    "new_clusters": len(new_clusters),
                    "new_relations": len(new_relations),
                    "cumulative_clusters": len(cumulative_clusters),
                    "cumulative_relations": len(cumulative_relations),
                    "cluster_novelty": round(novelty_c, 3),
                    "relation_novelty": round(novelty_r, 3),
                }
            )

    return {"runs": results}


def print_stability_table(stability: dict):
    print("\n" + "=" * 80)
    print("METRIC 3: Cross-run Stability & Novelty Decay")
    print("=" * 80)

    if "message" in stability:
        print(f"  {stability['message']}")
        return

    print(
        f"{'Run':<30} {'NewCl':>7} {'NewRel':>7} {'CumCl':>7} {'CumRel':>7} {'Nov.Cl':>8} {'Nov.Rel':>8}"
    )
    print("-" * 80)
    for r in stability["runs"]:
        print(
            f"{r['run']:<30} {r['new_clusters']:>7} {r['new_relations']:>7} "
            f"{r['cumulative_clusters']:>7} {r['cumulative_relations']:>7} "
            f"{r['cluster_novelty']:>8.3f} {r['relation_novelty']:>8.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Saturation Curve
# ─────────────────────────────────────────────────────────────────────────────


def compute_saturation(runs: list[dict]) -> list[dict]:
    """Track assertions per run and cumulative novelty."""
    curve = []
    for i, run in enumerate(runs):
        nf = run.get("novelty_filtering", {})
        total_proposed = sum(a.get("proposed", 0) for a in nf.values())
        total_novel = sum(a.get("novel", 0) for a in nf.values())
        total_filtered = sum(a.get("filtered_as_duplicates", 0) for a in nf.values())
        novelty_rate = (total_novel / total_proposed) if total_proposed > 0 else 0

        curve.append(
            {
                "run_n": i + 1,
                "run_id": run["_dir"],
                "proposed": total_proposed,
                "novel": total_novel,
                "filtered": total_filtered,
                "novelty_rate": round(novelty_rate, 3),
            }
        )
    return curve


def print_saturation_table(curve: list[dict]):
    print("\n" + "=" * 80)
    print("METRIC 4: Saturation Curve")
    print("=" * 80)
    print(f"{'Run#':>5} {'Proposed':>10} {'Novel':>10} {'Filtered':>10} {'Novelty%':>10}")
    print("-" * 50)
    for r in curve:
        print(
            f"{r['run_n']:>5} {r['proposed']:>10} {r['novel']:>10} "
            f"{r['filtered']:>10} {r['novelty_rate']*100:>9.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 5: Confidence Distribution Statistics
# ─────────────────────────────────────────────────────────────────────────────


def compute_confidence_stats(history: dict) -> dict:
    """Detailed confidence statistics per assertion type."""
    conf_by_type = defaultdict(list)

    for c in history.get("clusters", []):
        conf_by_type[f"cluster/{c['cluster_type']}"].append(c["confidence"])
    for r in history.get("relations", []):
        conf_by_type[f"relation/{r['relation_type']}"].append(r["confidence"])
    for p in history.get("paths", []):
        conf_by_type[f"path/{p.get('path_type', 'unknown')}"].append(p["confidence"])

    stats = {}
    for t, confs in sorted(conf_by_type.items()):
        n = len(confs)
        mean = sum(confs) / n
        sorted_c = sorted(confs)
        median = sorted_c[n // 2]
        stats[t] = {
            "n": n,
            "mean": round(mean, 3),
            "median": round(median, 3),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3),
            "std": round((sum((c - mean) ** 2 for c in confs) / n) ** 0.5, 3),
        }
    return stats


def print_confidence_stats(stats: dict):
    print("\n" + "=" * 85)
    print("METRIC 5: Confidence Distribution per Assertion Type")
    print("=" * 85)
    print(f"{'Type':<28} {'n':>5} {'Mean':>7} {'Median':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print("-" * 85)
    for t, s in stats.items():
        print(
            f"{t:<28} {s['n']:>5} {s['mean']:>7.3f} {s['median']:>7.3f} "
            f"{s['min']:>7.3f} {s['max']:>7.3f} {s['std']:>7.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 6: Confidence Calibration Plot (requires instrumented runs)
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_calibration(history: dict, output_path: Path):
    """
    Scatter plot: c_LLM vs c_val for all assertions.
    Requires metadata.llm_confidence and metadata.validation_score fields
    (added by instrumented runs).
    """
    if not HAS_MATPLOTLIB:
        print("⚠ Skipping calibration plot (matplotlib not installed)")
        return

    points = []  # (c_llm, c_val, type, agent)
    agent_colors = {
        "geographic": "#2196F3",
        "chronological": "#4CAF50",
        "typological": "#FF9800",
        "nearTo": "#2196F3",
        "contemporaryWith": "#4CAF50",
        "similarTo": "#FF9800",
    }

    for c in history.get("clusters", []):
        meta = c.get("metadata", {})
        if "llm_confidence" in meta and "validation_score" in meta:
            points.append(
                (
                    meta["llm_confidence"],
                    meta["validation_score"],
                    c["cluster_type"],
                )
            )

    for r in history.get("relations", []):
        meta = r.get("metadata", {})
        if "llm_confidence" in meta and "validation_score" in meta:
            points.append(
                (
                    meta["llm_confidence"],
                    meta["validation_score"],
                    r["relation_type"],
                )
            )

    for p in history.get("paths", []):
        meta = p.get("metadata", {})
        if "llm_confidence" in meta and "validation_score" in meta:
            points.append(
                (
                    meta["llm_confidence"],
                    meta["validation_score"],
                    p.get("path_type", "path"),
                )
            )

    if not points:
        print("\n⚠ No disaggregated confidence data found in knowledge history.")
        print("  Run an instrumented experiment to collect c_LLM and c_val separately.")
        print("  (See paper/reviews.md, Step 2)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Group by type for legend
    by_type = defaultdict(lambda: ([], []))
    for c_llm, c_val, atype in points:
        by_type[atype][0].append(c_llm)
        by_type[atype][1].append(c_val)

    for atype, (llm_vals, val_vals) in by_type.items():
        color = agent_colors.get(atype, "#9E9E9E")
        ax.scatter(llm_vals, val_vals, c=color, label=atype, alpha=0.7, s=50, edgecolors="white")

    # Diagonal (perfect agreement)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect agreement")

    # Penalty threshold zone
    ax.fill_between([0, 0.6], [0.4, 1.0], [0, 0.6], alpha=0.05, color="red")
    ax.fill_between([0.4, 1.0], [0, 0.6], [0.4, 1.0], alpha=0.05, color="red")

    ax.set_xlabel("LLM Confidence ($c_{LLM}$)", fontsize=12)
    ax.set_ylabel("Validation Score ($c_{val}$)", fontsize=12)
    ax.set_title("Hybrid Confidence Calibration", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Calibration plot saved to {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Saturation Curve
# ─────────────────────────────────────────────────────────────────────────────


def plot_saturation(curve: list[dict], output_path: Path):
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    runs = [r["run_n"] for r in curve]
    novel = [r["novel"] for r in curve]
    proposed = [r["proposed"] for r in curve]
    novelty_rate = [r["novelty_rate"] * 100 for r in curve]

    ax1.bar(runs, proposed, alpha=0.3, label="Proposed", color="#90CAF9", width=0.6)
    ax1.bar(runs, novel, alpha=0.8, label="Accepted (novel)", color="#1976D2", width=0.6)
    ax1.set_xlabel("Run #", fontsize=12)
    ax1.set_ylabel("Assertions", fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(runs, novelty_rate, "r-o", label="Novelty rate %", linewidth=2)
    ax2.set_ylabel("Novelty Rate (%)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Knowledge Saturation Across Runs", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saturation plot saved to {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Confidence Distribution
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_distribution(history: dict, output_path: Path):
    if not HAS_MATPLOTLIB:
        return

    confs_by_category = defaultdict(list)
    for c in history.get("clusters", []):
        confs_by_category[c["cluster_type"]].append(c["confidence"])
    for r in history.get("relations", []):
        confs_by_category[r["relation_type"]].append(r["confidence"])
    for p in history.get("paths", []):
        confs_by_category[p.get("path_type", "path")].append(p["confidence"])

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = sorted(confs_by_category.keys())
    data = [confs_by_category[l] for l in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#795548"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0.70, color="green", linestyle="--", alpha=0.5, label="Validated (≥0.70)")
    ax.axhline(y=0.60, color="orange", linestyle="--", alpha=0.5, label="PendingReview (≥0.60)")

    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_title("Confidence Distribution by Assertion Type", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.4, 1.05)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confidence distribution plot saved to {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Export structured results for LaTeX
# ─────────────────────────────────────────────────────────────────────────────


def export_results(
    novelty: dict,
    tiers: dict,
    stability: dict,
    saturation: list[dict],
    conf_stats: dict,
    output_path: Path,
):
    results = {
        "novelty_rates": novelty,
        "confidence_tiers": tiers,
        "stability": stability,
        "saturation": saturation,
        "confidence_stats": conf_stats,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Structured results saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("xch-MIND — Quantitative Metrics Analysis")
    print("=" * 50)

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    runs = load_metadata_runs()
    history = load_knowledge_history()

    print(f"\nLoaded {len(runs)} runs with metadata")
    print(
        f"Knowledge history: {len(history.get('clusters', []))} clusters, "
        f"{len(history.get('relations', []))} relations, "
        f"{len(history.get('paths', []))} paths"
    )

    # Compute metrics
    novelty = compute_novelty_rates(runs)
    print_novelty_table(novelty)

    tiers = compute_confidence_tiers(history)
    print_tier_table(tiers)

    stability = compute_jaccard_stability(history)
    print_stability_table(stability)

    saturation = compute_saturation(runs)
    print_saturation_table(saturation)

    conf_stats = compute_confidence_stats(history)
    print_confidence_stats(conf_stats)

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_saturation(saturation, RESULTS_DIR / "saturation_curve.png")
        plot_confidence_distribution(history, RESULTS_DIR / "confidence_distribution.png")
        plot_confidence_calibration(history, RESULTS_DIR / "confidence_calibration.png")

    # Export structured data
    export_results(novelty, tiers, stability, saturation, conf_stats, RESULTS_DIR / "metrics.json")

    print("\n✓ Done! Results in paper/metrics/")


if __name__ == "__main__":
    main()
