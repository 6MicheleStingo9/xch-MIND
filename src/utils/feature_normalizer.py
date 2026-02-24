"""
Feature Label Normalizer — Canonical label normalization for extracted features.

Combines syntactic normalization (lowercase, hyphens, whitespace) with SpaCy
English lemmatization to produce stable canonical forms, and similarity-based
deduplication to merge near-duplicate labels across LLM runs.

Design rationale
----------------
The LLM (Gemini) extracts feature labels as free-text strings.  Across runs it
may produce surface variants of the same concept ("stones" vs "stone",
"peri-urban" vs "periurban", "memory of ancestors" vs "ancestor memory").
These variants cause feature_entities keys to diverge, inflating the total
assertion count and reducing cross-run reproducibility.

The normalizer operates in two stages:

1. **Syntactic + Lemma normalization** (deterministic, per-label):
   lowercase → strip → hyphens/underscores → spaces → SpaCy EN lemmatization

2. **Similarity-based deduplication** (pairwise, per-category):
   Within the same feature category, labels whose cosine similarity (on SpaCy
   EN word vectors) exceeds a configurable threshold are merged under a single
   canonical label (the one shared by the most entities).
"""

import logging
import re
from typing import Any

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpaCy EN model (lazy loaded, separate from the IT model in
# semantic_similarity.py which is used for description-level similarity)
# ---------------------------------------------------------------------------
_nlp_en: Language | None = None


def _get_en_model() -> Language:
    """Lazy-load the SpaCy English model."""
    global _nlp_en
    if _nlp_en is None:
        logger.info("Loading SpaCy model 'en_core_web_lg' for feature normalisation...")
        _nlp_en = spacy.load("en_core_web_lg")
        logger.info("SpaCy EN model loaded (vectors: %s)", _nlp_en.vocab.vectors.shape)
    return _nlp_en


# ---------------------------------------------------------------------------
# Stage 1 — Syntactic + Lemma normalization
# ---------------------------------------------------------------------------


def normalize_feature_label(label: str) -> str:
    """
    Produce a canonical form for a feature label.

    Pipeline:
        1. lowercase + strip
        2. hyphens / underscores → spaces  (peri-urban → peri urban)
        3. collapse whitespace
        4. SpaCy EN lemmatization          (stones → stone)
        5. re-sort tokens alphabetically   (memory of ancestor → ancestor memory of)
           — disabled: preserves natural word order for readability.

    Args:
        label: Raw feature label from the LLM.

    Returns:
        Normalised label string.
    """
    # Step 1-3: purely syntactic
    s = label.strip().lower()
    s = re.sub(r"[_\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return s

    # Step 3b: domain-specific canonical forms for OOV compound words
    _canonical: dict[str, str] = {
        "periurban": "peri urban",
        "piedritti": "piedritto",
    }
    s = _canonical.get(s, s)

    # Step 4: lemmatisation
    nlp = _get_en_model()
    doc = nlp(s)
    lemmatized = " ".join(tok.lemma_ for tok in doc)

    return lemmatized


# ---------------------------------------------------------------------------
# Stage 2 — Similarity-based deduplication
# ---------------------------------------------------------------------------


def deduplicate_feature_labels(
    feature_entities: dict[tuple[str, str], list[tuple[str, str]]],
    similarity_threshold: float = 0.85,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Merge feature groups whose labels are near-duplicates (same category).

    For each pair of labels within the same category, if SpaCy cosine
    similarity ≥ *similarity_threshold*, their entity lists are merged under
    the label that has the most entities (canonical winner).

    Uses Union-Find to handle transitive merges (A≈B, B≈C → all merge).

    Args:
        feature_entities: Mapping (category, normalised_label) → [(uri, name)].
        similarity_threshold: Cosine similarity threshold for merging.

    Returns:
        A new dict with near-duplicate groups merged.
    """
    nlp = _get_en_model()

    # Group keys by category
    by_category: dict[str, list[tuple[str, str]]] = {}
    for key in feature_entities:
        cat, _label = key
        by_category.setdefault(cat, []).append(key)

    # Union-Find helpers
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])  # path compression
            x = parent[x]
        return x

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the one with more entities as root
            if len(feature_entities.get(ra, [])) >= len(feature_entities.get(rb, [])):
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Pairwise similarity within each category
    merge_count = 0
    for cat, keys in by_category.items():
        if len(keys) < 2:
            continue

        # Pre-compute docs for this category's labels
        docs: dict[tuple[str, str], Any] = {}
        for key in keys:
            _, label = key
            docs[key] = nlp(label)

        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1 :]:
                doc_a = docs[key_a]
                doc_b = docs[key_b]

                # Skip if either has zero vector (OOV)
                if doc_a.vector_norm == 0 or doc_b.vector_norm == 0:
                    continue

                sim = doc_a.similarity(doc_b)
                if sim >= similarity_threshold:
                    logger.info(
                        "FeatureNormalizer: Merging '%s' ↔ '%s' (category=%s, sim=%.3f)",
                        key_a[1],
                        key_b[1],
                        cat,
                        sim,
                    )
                    union(key_a, key_b)
                    merge_count += 1

    if merge_count == 0:
        logger.info("FeatureNormalizer: No near-duplicate labels found to merge")
        return feature_entities

    # Rebuild merged dict
    merged: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, entities in feature_entities.items():
        root = find(key)
        if root not in merged:
            merged[root] = []
        # Extend, but deduplicate by URI
        existing_uris = {uri for uri, _ in merged[root]}
        for uri, name in entities:
            if uri not in existing_uris:
                merged[root].append((uri, name))
                existing_uris.add(uri)

    logger.info(
        "FeatureNormalizer: Merged %d label pairs → %d groups (was %d)",
        merge_count,
        len(merged),
        len(feature_entities),
    )

    return merged
