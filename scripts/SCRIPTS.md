# Utility Scripts

This directory contains utility scripts for maintenance, diagnostics, analysis, and testing of the xch-MIND system.

## Available Scripts

### 1. `analyze_cache.py`

Analyzes the completeness of geographic data (coordinates) for the entities in the `entities/` folder by checking the `sparql_cache.json`.

- **Usage**: `python scripts/analyze_cache.py`
- **Output**: Generates a report in `output/cache_analysis_report.json`.

### 2. `check_models.py`

Lists the available LLM models from the configured provider (e.g., Google Gemini) that support content generation.

- **Usage**: `python scripts/check_models.py`

### 3. `test_logging.py`

A demonstration script to verify the custom colored logging system and agent-specific icons.

- **Usage**: `python scripts/test_logging.py`

### 4. `compute_metrics.py`

Computes quantitative metrics from completed pipeline runs for paper evaluation. Loads all `metadata.json` files in `output/` and the accumulated `.knowledge_history.json` to produce:

- Validation pass rate per agent
- Assertion distribution across confidence tiers (Validated ≥0.70 / PendingReview 0.60–0.69 / Rejected <0.60)
- Cross-run Jaccard stability analysis
- Saturation curve (cumulative assertions per run)
- Confidence calibration plot (c_LLM vs c_val)

- **Usage**: `python scripts/compute_metrics.py`
- **Output**: Tables to stdout; plots saved to `paper/metrics/` (`saturation_curve.png`, `confidence_distribution.png`, `metrics.json`).
- **Requirements**: `matplotlib` (optional — tables are printed regardless).

### 5. `baseline_single_prompt.py`

Implements the single-prompt baseline for comparison with the xch-MIND multi-agent approach. Sends all 53 entities to the LLM in **one prompt** requesting geographic, chronological, and typological clusters plus thematic paths in a single structured JSON response. No validation, no cross-run memory, no multi-agent decomposition.

- **Usage**: `python scripts/baseline_single_prompt.py`
- **Output**: Prints structured JSON response and saves `baseline_result.json` in `output/`.
- **Note**: Uses the provider and model configured in `src/config/config.yaml`. Rate limiting applies.

### 6. `ablation_comparison.py`

Compares the standard xch-MIND run (with hybrid validation, α=0.7/β=0.3) against the no-validation ablation run (confidence score = raw LLM output only). Loads `.knowledge_history.json` from both `output/` and `output_ablation_noval/` directories.

Produces:
- Side-by-side summary table (assertion count, confidence mean/std, tier distribution)
- Confidence distribution histogram figure saved to `paper/metrics/ablation_comparison.png`

- **Usage**: `python scripts/ablation_comparison.py`
- **Prerequisites**: Both `output/.knowledge_history.json` and `output_ablation_noval/.knowledge_history.json` must exist (i.e., standard and ablation runs must have been executed).
- **Requirements**: `matplotlib`, `numpy`.

---

## How to run

All scripts should be run from the **project root** directory to ensure paths are resolved correctly.

```bash
python scripts/<script_name>.py
```
