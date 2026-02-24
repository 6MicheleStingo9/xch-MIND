# Utility Scripts

This directory contains utility scripts for maintenance, diagnostics, and testing of the xch-MIND system.

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

---

## How to run

All scripts should be run from the **project root** directory to ensure paths are resolved correctly.

```bash
python scripts/<script_name>.py
```
