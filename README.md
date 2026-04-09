# xch-MIND: Multi-agent Interpretive Nexus Discovery

## A Case Study with Hybrid LLM-Algorithmic Validation on Cultural Heritage Data

### Hybrid Clustering and Semantic Enrichment Across Multiple Dimensions

## Overview

This project demonstrates how to build interpretive layers over institutional Linked Data (ArCo ontology) without modifying the source, using autonomous LLM agents to generate enriched Knowledge Graphs for cultural heritage domain.

**Case Study:** Creating a "Virtual Archaeological Park" from 53 dolmen entities catalogued with ArCo ontology.

## Architecture

- **Source Layer:** ArCo ontology + 53 dolmen entities (immutable XML, parsed and resolved via SPARQL where possible)
- **Interpretive Layer:** xch: namespace with inferred relationships, clusters, and thematic paths
- **Multi-Agent System:** Orchestrator + specialized workers (Geospatial, Chronological, Typological, Narrative, Validation)
- **Technology Stack:**
  - **LLM Providers:** Google Gemini (2.5 Flash) or Ollama (open-source local models), configurable via `src/config/config.yaml`
  - **Graph Framework:** LangGraph for multi-agent orchestration
  - **RDF Layer:** RDFLib for Turtle serialization and JSON-LD generation
  - **NLP:** SpaCy (Italian models, `it_core_news_lg`) for semantic similarity in typological clustering
  - **Config Management:** Pydantic v2 Settings for YAML-based configuration
  - **Rate Limiting:** Persistent rate limiter with configurable limits per LLM tier (Gemini free tier: 15 RPM / 20 RPD; Ollama: unlimited)

## Installation

### Prerequisites

- Python 3.10+
- Google Gemini API key (or Ollama for local models)

### Setup

```bash
# Clone repository
git clone https://github.com/6MicheleStingo9/xch-MIND.git
cd xch-MIND

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Edit .env and add your GOOGLE_API_KEY
```

## Configuration

Configuration is managed through `src/config/config.yaml`, loaded via Pydantic Settings in `src/config/settings.py`.

### Quick Start

```bash
# Run with default config (53 dolmen entities)
python -m src.main

# Run with entity limit (useful for testing)
python -m src.main --limit 5

# Enable verbose logging and debug traces
python -m src.main --verbose --debug
```

### Output Format

Generated knowledge graphs are produced in:

- **Turtle** (`.ttl`) — RDF standard format
- **JSON-LD** (`.jsonld`) — Linked Data JSON format

Both contain the same assertions with `xch:` namespace enrichments.

## Operational Strategy: Incremental Memory-Based Multi-Run

This system is designed for **iterative execution** as an architectural pattern (not optional):

### How It Works

1. **Cross-Run Memory:** Each pipeline execution reads from `.knowledge_history.json`, which persists all assertions discovered in previous runs.
2. **Deduplication:** Agents generate candidate assertions, but the system filters out duplicates before adding them to the knowledge graph.
3. **Rate Limiting:** State is persisted in `.rate_limit.json` to track API consumption (configured per provider/tier in `config.yaml`):
   - **Gemini free tier:** 15 requests/minute, 20 requests/day
   - **Ollama:** Unlimited (local execution, no API constraints)
4. **Diminishing Returns by Design:** Typological analysis often saturates quickly with small datasets (26/53 entities have rich descriptions), while geographic and chronological analysis may yield novel assertions on subsequent runs due to LLM stochasticity.

### Example Workflow

```bash
# Run 1: Discover initial assertions
python -m src.main

# Check output/
ls -la output/*.ttl
cat output/.knowledge_history.json

# Run 2: Discover incremental assertions (queries LLM again, deduplicates)
python -m src.main

# Repeat as needed — knowledge accumulates, redundancy is filtered
```

### Why Incremental by Design?

The multi-run architecture is **structural to the system** (not optional), enabling:

- **Progressive knowledge accumulation** across runs without redundancy
- **Cross-run memory** allowing smart deduplication
- **Graceful handling** of rate limits (Gemini free tier: 15 RPM forces batching; paid tiers and Ollama allow more aggressive scheduling)
- **LLM stochasticity tolerance:** Multiple runs can discover diverse valid interpretations while filtering duplicates

## Logging

The execution output is enhanced with colored logs and agent-specific icons to track the multi-agent workflow:

- 🤖 **Orchestrator**: Routing and goal selection
- 📍 **GeoAnalyzer**: Geographic clustering findings
- ⏳ **TemporalAnalyzer**: Chronological normalization and clusters
- 🏛️ **TypeAnalyzer**: Architectural feature extraction
- 🗺️ **PathGenerator**: Thematic itinerary synthesis
- 🔗 **TripleGenerator**: Assertions to RDF mapping
- ✅ **TripleValidator**: Final Knowledge Graph validation

## Documentation

- [Ontology Evolution Guide](docs/ONTOLOGY_EVOLUTION.md): How to extend the system with new properties and relations.

## Project Structure

```
xch-MIND/
├── docs/
│   └── ONTOLOGY_EVOLUTION.md        # Guide for extending the ontology
├── entities/                        # 53 dolmen XML files (ArCo format)
├── ontology/
│   ├── ArCo.owl                    # ArCo ontology (immutable source)
│   └── xch/
│       └── xch-core.ttl            # xch-MIND interpretive ontology
├── output/                         # Generated knowledge graphs (TTL, JSON-LD)
├── scripts/                        # Utility scripts
│   ├── analyze_cache.py            # Geographic data completeness report
│   ├── check_models.py             # List available LLM models
│   ├── test_logging.py             # Verify colored logging system
│   ├── compute_metrics.py          # Quantitative metrics for paper evaluation
│   ├── baseline_single_prompt.py   # Single-prompt baseline comparison
│   └── ablation_comparison.py      # No-validation ablation comparison
├── src/
│   ├── agents/                     # Multi-agent system
│   │   ├── base.py                # BaseAgent abstract class
│   │   ├── orchestrator.py        # Orchestrator agent (routing)
│   │   ├── models.py              # Pydantic models for agent I/O
│   │   ├── confidence.py          # Hybrid confidence scoring (70/30)
│   │   ├── validation.py          # Algorithmic validation functions
│   │   └── workers/               # Specialized worker agents
│   │       ├── geo_analyzer.py    # Geographic clustering & proximity
│   │       ├── temporal_analyzer.py # Period normalization & relations
│   │       ├── type_analyzer.py   # Typological feature extraction
│   │       └── path_generator.py  # Thematic path generation
│   ├── config/
│   │   ├── config.yaml            # Configuration (LLM, agents, workflow)
│   │   └── settings.py            # Pydantic Settings loader
│   ├── llm/
│   │   ├── provider.py            # LLM provider abstraction (Gemini/Ollama)
│   │   └── __init__.py
│   ├── loaders/
│   │   └── arco_loader.py         # XML entity parsing and SPARQL resolution
│   ├── triples/
│   │   ├── generator.py           # Assertions → RDF triples mapping
│   │   ├── serializer.py          # Graph serialization (Turtle, JSON-LD)
│   │   └── validator.py           # RDF validation & consistency checking
│   ├── utils/
│   │   ├── logging.py             # Colored logging with agent icons
│   │   ├── rate_limiter.py        # Rate limiting with state persistence
│   │   └── __init__.py
│   ├── workflow/
│   │   ├── graph.py               # LangGraph StateGraph definition
│   │   ├── conditions.py          # Routing conditions
│   │   └── nodes.py               # Node functions for agents
│   ├── pipeline.py                # Main pipeline orchestrator
│   ├── main.py                    # CLI entry point
│   └── __init__.py
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Agent Details

### **Orchestrator (🤖)**

Routes entities to specialized agents based on entity type and availability of descriptive data. Maintains comprehensive vs. focused mode logic.

### **GeoAnalyzer (📍)**

- Clusters dolmen by geographic proximity (default: 10 km threshold)
- Validates clusters using SPARQL distance queries where available
- Generates `xch:nearTo` relations with confidence scores
- **Parameters (config.yaml):** `max_clusters=15`, `max_pairs=20`

### **TemporalAnalyzer (⏳)**

- Normalizes temporal metadata (centuries, date ranges, uncertainties)
- Clusters entities by chronological proximity
- Generates `xch:contemporaryWith` relations
- **Parameters:** `max_clusters=10`, `max_pairs=15`

### **TypeAnalyzer (🏛️)**

- Extracts architectural features via LLM + SpaCy semantic similarity
- Clusters entities by typological similarity (dolmen subtypes, structural features)
- Generates `xch:similarTo` relations with dual validation (LLM confidence + embedding similarity)
- **Parameters:** `max_clusters=10`, `max_pairs=20`, `embedding_similarity_threshold=0.70`
- **⚠️ Limitation:** Requires entity descriptions; 26/53 entities lack rich `description_text`, limiting typological coverage

### **PathGenerator (🗺️)**

Synthesizes thematic itineraries through entity clusters (geographic routes, chronological sequences, typological themes).

### **TripleGenerator & TripleValidator (🔗 ✅)**

Converts agent assertions to RDF triples with provenance metadata; validates consistency and grounding to source ArCo entities.

## Output Structure

Generated knowledge graphs (in `output/<timestamp>/`) include:

### Files

- **`knowledge_graph.ttl`** — RDF Turtle format with all xch: assertions
- **`knowledge_graph.jsonld`** — JSON-LD serialization (same data, different format)
- **`metadata.json`** — Execution summary (assertion count, triple count, runtime, LLM calls, error log)
- **`execution.log`** — Detailed agent execution trace with colors and icons
- **`.knowledge_history.json`** (root `output/`) — Cross-run persistent memory of all discovered clusters and relations
- **`.rate_limit.json`** (root `output/`) — API consumption tracking for rate limit compliance

### Triple Structure

Each assertion triple includes:

```turtle
xch:Entity_A xch:nearTo xch:Entity_B ;
    xch:confidence "0.92"^^xsd:float ;
    xch:generated_by "GeoAnalyzer"@en ;
    xch:grounded_in arco:Entity_A_ArCo_URI ;
    xch:validated "true"^^xsd:boolean .
```

- **Provenance:** `xch:generated_by` identifies the agent
- **Confidence:** Hybrid scoring (70% LLM confidence + 30% algorithmic validation)
- **Grounding:** Links back to original ArCo entities for auditability
- **Validation:** Boolean flag indicating whether assertion passed consistency checks

## Known Limitations

### Data-Driven Constraints

1. **Typological Saturation (26/53 entities)**
   - Only 27 dolmen have rich `description_text` in source XML
   - TypeAnalyzer cannot generate meaningful typological relations for entities with only name/location
   - Recommendation: Enrich entity descriptions or use category-only analysis

2. **Geographic Threshold (10 km)**
   - Some valid clusters (e.g., Gargano plateau, 12.9 km span) are rejected by default `clustering_threshold`
   - Recommendation: Test with `geospatial.clustering_threshold: 15` for regional analysis

3. **Chronological Uncertainties**
   - Entities with broad date ranges (e.g., "4th-2nd millennium BCE") may not cluster as expected
   - Validation rejects pairs with >500 year gaps (configurable via `chronological.max_temporal_distance`)

### Operational Constraints

4. **LLM Rate Limits (Provider & Tier Dependent)**
   - **Gemini free tier:** 15 requests/minute, 20 requests/day (default in `config.yaml`)
   - **Gemini Tier 1+:** 2000 requests/minute, unlimited daily (update `rate_limiting.enabled: false` in `config.yaml` or increase limits)
   - **Ollama:** Unlimited (local execution, no API constraints; set `llm.provider: "ollama"` in `config.yaml`)
   - Full 53-entity pipeline requires ~4 LLM calls; batch size depends on tier and provider
   - State persists in `.rate_limit.json` across runs

5. **Stochastic LLM Behavior (Gemini & Ollama)**
   - LLMs may propose different clusters on re-run due to sampling temperature
   - Cross-run memory deduplicates, but new valid assertions may emerge across multiple runs
   - Design choice: Accept variance for diversity over determinism

## Contributing

This is a research prototype for academic publication. Contributions welcome after initial paper submission.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

**Note:** The ArCo ontology (`ontology/ArCo.owl`) is licensed separately by ICCDU/CNR. The xch-MIND ontology (`ontology/xch/xch-core.ttl`) is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this system in your research, please cite:

**Conference paper:**
```bibtex
@inproceedings{stingo2026xchmind,
  title     = {xch-MIND: Multi-Agent Interpretive Knowledge Graph Augmentation over Cultural Heritage Linked Data},
  author    = {Stingo, Michele},
  booktitle = {Proceedings of the 5th Workshop on LLM-Integrated Knowledge Graph Generation from Text (TEXT2KG) at ESWC 2026},
  year      = {2026},
  address   = {Dubrovnik, Croatia},
  month     = {May}
}
```

**Software:**
```bibtex
@software{xch-mind2026,
  title = {xch-MIND: Multi-Agent Interpretive Knowledge Graph Augmentation over Cultural Heritage Linked Data},
  author = {Stingo, Michele},
  year = {2026},
  url = {https://github.com/6MicheleStingo9/xch-MIND}
}
```

## Contact

For questions, issues, or collaboration opportunities:

- **Project Lead:** Michele Stingo ([michele.stingo@activadigital.it](mailto:michele.stingo@activadigital.it))
- **Repository:** [GitHub - xch-MIND](https://github.com/6MicheleStingo9/xch-MIND)
- **Issues & Discussions:** Please use the GitHub Issues tab for bug reports and feature requests
