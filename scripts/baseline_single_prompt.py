#!/usr/bin/env python3
"""
Baseline: single-prompt approach for comparison with xch-MIND multi-agent.

Sends ALL 53 entities to the LLM in ONE prompt asking for geographic,
chronological, typological clusters + paths in a single response.
No validation, no memory, no multi-agent decomposition.
"""

import json
import sys
import time
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.loaders.arco_loader import load_all_entities
from src.llm.provider import create_provider
from src.utils.rate_limiter import RateLimiter
from src.config.settings import load_config


# ── Pydantic schema for single-prompt response ───────────────────────


class ClusterProposal(BaseModel):
    cluster_type: str = Field(description="One of: geographic, chronological, typological")
    label: str
    member_names: list[str]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class RelationProposal(BaseModel):
    relation_type: str = Field(description="One of: nearTo, contemporaryWith, similarTo")
    source_name: str
    target_name: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class PathProposal(BaseModel):
    title: str
    theme: str
    stops: list[str] = Field(description="Ordered list of site names")
    narrative: str
    confidence: float = Field(ge=0.0, le=1.0)


class SinglePromptResponse(BaseModel):
    clusters: list[ClusterProposal]
    relations: list[RelationProposal]
    paths: list[PathProposal]
    overall_observations: str


# ── System and Human prompts ─────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert archaeologist specializing in megalithic monuments.
Your task is to analyze a collection of archaeological sites and produce a comprehensive
interpretive analysis covering geographic, chronological, and typological dimensions,
plus thematic visit paths.

OUTPUT: Respond with structured JSON matching the SinglePromptResponse schema.
"""

HUMAN_PROMPT = """\
Analyze the following {total} archaeological sites and produce ALL of the following:

SITE DATA:
{sites_json}

1. GEOGRAPHIC CLUSTERS (max 15): Group sites sharing territorial context (same municipality, valley, region).
2. CHRONOLOGICAL CLUSTERS (max 10): Group sites from the same period/cultural phase.
3. TYPOLOGICAL CLUSTERS (max 10): Group sites sharing architectural or functional features.
4. NEAR-TO RELATIONS (max 20): Pairs of nearby sites with meaningful spatial connections.
5. CONTEMPORARY-WITH RELATIONS (max 15): Pairs of sites from the same period.
6. SIMILAR-TO RELATIONS (max 20): Pairs of typologically similar sites.
7. THEMATIC PATHS (max 5): Ordered visit sequences with narrative.

For each item, provide confidence (0-1) and reasoning.
Focus on quality over quantity. Use ONLY the provided data.
"""


def build_site_summary(entity) -> dict:
    """Build compact JSON summary for each entity."""
    return {
        "name": entity.display_name,
        "coordinates": (
            {"lat": entity.latitude, "lon": entity.longitude}
            if entity.latitude and entity.longitude
            else None
        ),
        "municipality": entity.municipality,
        "region": entity.region_name,
        "period": entity.period_label,
        "category": entity.category,
        "description": (entity.description or "")[:200],
        "historical_info": (entity.historical_info or "")[:150],
    }


def main():
    print("=" * 60)
    print("BASELINE: Single-Prompt Analysis")
    print("=" * 60)

    # Load settings and entities
    settings = load_config(str(ROOT / "src" / "config" / "config.yaml"))
    entities = load_all_entities(str(ROOT / "entities"))
    print(f"Loaded {len(entities)} entities")

    # Build site summaries
    sites = [build_site_summary(e) for e in entities]
    sites_json = json.dumps(sites, indent=2, ensure_ascii=False)

    # Create LLM provider with rate limiter
    llm_cfg = settings.llm
    provider_name = llm_cfg.provider
    if provider_name == "gemini":
        model_name = llm_cfg.gemini.model
        temperature = llm_cfg.gemini.temperature
    else:
        model_name = llm_cfg.ollama.model
        temperature = llm_cfg.ollama.temperature

    rate_limiter = None
    if llm_cfg.rate_limiting.enabled:
        rate_limiter = RateLimiter(
            requests_per_minute=llm_cfg.rate_limiting.requests_per_minute,
            requests_per_day=llm_cfg.rate_limiting.requests_per_day,
            max_retries=llm_cfg.rate_limiting.max_retries,
            base_delay=llm_cfg.rate_limiting.base_delay,
        )

    provider = create_provider(
        provider_type=provider_name,
        model_name=model_name,
        temperature=temperature,
        rate_limiter=rate_limiter,
    )

    # Build prompt
    human = HUMAN_PROMPT.format(total=len(entities), sites_json=sites_json)

    print(f"\nPrompt size: {len(SYSTEM_PROMPT) + len(human):,} chars")
    print(f"Calling LLM ({model_name})...")
    print()

    # Single LLM call
    t0 = time.time()
    try:
        response = provider.invoke_structured(
            system_prompt=SYSTEM_PROMPT,
            human_prompt=human,
            output_schema=SinglePromptResponse,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"LLM call completed in {elapsed:.1f}s")

    # ── Analyze results ──────────────────────────────────────────────
    clusters_by_type = {}
    for c in response.clusters:
        clusters_by_type.setdefault(c.cluster_type, []).append(c)

    relations_by_type = {}
    for r in response.relations:
        relations_by_type.setdefault(r.relation_type, []).append(r)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Clusters
    total_clusters = len(response.clusters)
    print(f"\nClusters: {total_clusters}")
    for ctype, items in sorted(clusters_by_type.items()):
        confs = [c.confidence for c in items]
        print(f"  {ctype}: {len(items)} (mean conf: {sum(confs)/len(confs):.3f})")
        for c in items:
            members_str = ", ".join(c.member_names[:4])
            if len(c.member_names) > 4:
                members_str += f"... (+{len(c.member_names)-4})"
            print(f"    - {c.label} [{len(c.member_names)} members, conf={c.confidence:.2f}]")
            print(f"      {members_str}")

    # Relations
    total_relations = len(response.relations)
    print(f"\nRelations: {total_relations}")
    for rtype, items in sorted(relations_by_type.items()):
        confs = [r.confidence for r in items]
        print(f"  {rtype}: {len(items)} (mean conf: {sum(confs)/len(confs):.3f})")

    # Paths
    print(f"\nPaths: {len(response.paths)}")
    for p in response.paths:
        print(f"  - {p.title} [{len(p.stops)} stops, conf={p.confidence:.2f}]")

    # Overall confidence
    all_confs = (
        [c.confidence for c in response.clusters]
        + [r.confidence for r in response.relations]
        + [p.confidence for p in response.paths]
    )
    print(f"\nOverall: {len(all_confs)} proposals")
    if all_confs:
        print(f"  Mean confidence: {sum(all_confs)/len(all_confs):.3f}")
        print(f"  Min: {min(all_confs):.3f}, Max: {max(all_confs):.3f}")
        validated = sum(1 for c in all_confs if c >= 0.70)
        print(f"  Validated (>=0.70): {validated} ({validated/len(all_confs)*100:.1f}%)")

    # Count total assertions (comparable to xch-MIND)
    total_assertions = total_clusters + total_relations + len(response.paths)
    # Add path stops
    total_stops = sum(len(p.stops) for p in response.paths)
    total_assertions += total_stops

    print(f"\n  Total assertion-level items: {total_assertions}")
    print(
        f"    (clusters: {total_clusters}, relations: {total_relations}, "
        f"paths: {len(response.paths)}, path stops: {total_stops})"
    )

    # ── Save results ─────────────────────────────────────────────────
    out_dir = ROOT / "output_baseline"
    out_dir.mkdir(exist_ok=True)

    result = {
        "model": model_name,
        "duration_seconds": round(elapsed, 2),
        "llm_calls": 1,
        "prompt_chars": len(SYSTEM_PROMPT) + len(human),
        "clusters": {
            "total": total_clusters,
            "by_type": {k: len(v) for k, v in clusters_by_type.items()},
            "mean_confidence": sum(c.confidence for c in response.clusters)
            / max(total_clusters, 1),
        },
        "relations": {
            "total": total_relations,
            "by_type": {k: len(v) for k, v in relations_by_type.items()},
            "mean_confidence": sum(r.confidence for r in response.relations)
            / max(total_relations, 1),
        },
        "paths": {
            "total": len(response.paths),
            "total_stops": total_stops,
            "mean_confidence": sum(p.confidence for p in response.paths)
            / max(len(response.paths), 1),
        },
        "overall": {
            "total_assertions": total_assertions,
            "mean_confidence": sum(all_confs) / max(len(all_confs), 1) if all_confs else 0,
            "min_confidence": min(all_confs) if all_confs else 0,
            "max_confidence": max(all_confs) if all_confs else 0,
            "validated_pct": (
                sum(1 for c in all_confs if c >= 0.70) / max(len(all_confs), 1) * 100
                if all_confs
                else 0
            ),
        },
        "raw_response": response.model_dump(),
    }

    out_path = out_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
