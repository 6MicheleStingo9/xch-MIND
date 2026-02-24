# Full Learning System

## Overview

FLS represents the natural evolution of the current memory system towards a **system that learns and improves over time**. While the present system only stores assertions and filters duplicates, FLS introduces learning mechanisms that enable the system to:

1. **Learn from validation feedback** - understand what works and what doesn't
2. **Self-calibrate parameters** - adapt thresholds based on data
3. **Correct LLM bias** - compensate for systematic tendencies in confidence scores
4. **Adapt prompts** - modify instructions based on success patterns

---

## Proposed Architecture

```
src/memory/
├── __init__.py
├── history_store.py          # ✅ Already implemented
├── novelty_detector.py       # ✅ Already implemented
├── context_builder.py        # ✅ Already implemented
├── feedback_tracker.py       # 🆕 Tracks successes/failures
├── pattern_learner.py        # 🆕 Extracts patterns from data
└── confidence_adjuster.py    # 🆕 Calibrates confidence scores
```

---

## Components to Implement

### 1. Feedback Tracker (`feedback_tracker.py`)

Tracks LLM proposals and validation results for each worker.

```python
"""
Feedback Tracker - Tracks validation successes and failures.
"""

from pydantic import BaseModel, Field
from typing import Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class WorkerFeedback(BaseModel):
    """Aggregated feedback for a worker."""

    worker_type: str
    proposals_total: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0

    # Rejection reasons with counts
    rejection_reasons: dict[str, int] = Field(default_factory=dict)

    # Confidence accuracy
    confidence_predictions: list[tuple[float, bool]] = Field(default_factory=list)
    # List of (predicted_confidence, accepted_or_not)

    @property
    def acceptance_rate(self) -> float:
        if self.proposals_total == 0:
            return 0.0
        return self.proposals_accepted / self.proposals_total

    def record_proposal(
        self,
        accepted: bool,
        confidence: float,
        rejection_reason: str | None = None,
    ) -> None:
        """Record the outcome of a proposal."""
        self.proposals_total += 1
        self.confidence_predictions.append((confidence, accepted))

        if accepted:
            self.proposals_accepted += 1
        else:
            self.proposals_rejected += 1
            if rejection_reason:
                self.rejection_reasons[rejection_reason] = (
                    self.rejection_reasons.get(rejection_reason, 0) + 1
                )


class FeedbackStore(BaseModel):
    """Complete feedback store."""

    version: str = "1.0"
    workers: dict[str, WorkerFeedback] = Field(default_factory=dict)

    # Feedback by assertion type
    assertion_feedback: dict[str, WorkerFeedback] = Field(default_factory=dict)


class FeedbackTracker:
    """
    Tracks validation feedback for learning.

    Usage:
        tracker = FeedbackTracker(output_dir)
        tracker.record_validation_result(
            worker="geo_analyzer",
            assertion_type="geographic_cluster",
            accepted=True,
            confidence=0.85,
            rejection_reason=None,
        )

        stats = tracker.get_worker_stats("geo_analyzer")
        print(f"Acceptance rate: {stats.acceptance_rate:.2%}")
    """

    DEFAULT_FILENAME = ".feedback_history.json"

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.feedback_path = self.output_dir / self.DEFAULT_FILENAME
        self._store: FeedbackStore | None = None

    @property
    def store(self) -> FeedbackStore:
        if self._store is None:
            self._store = self._load()
        return self._store

    def _load(self) -> FeedbackStore:
        if self.feedback_path.exists():
            try:
                with open(self.feedback_path, "r") as f:
                    data = json.load(f)
                return FeedbackStore(**data)
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")
        return FeedbackStore()

    def save(self) -> None:
        if self._store is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_path, "w") as f:
            json.dump(self._store.model_dump(), f, indent=2)

    def record_validation_result(
        self,
        worker: str,
        assertion_type: str,
        accepted: bool,
        confidence: float,
        rejection_reason: str | None = None,
    ) -> None:
        """Record a validation result."""
        # By worker
        if worker not in self.store.workers:
            self.store.workers[worker] = WorkerFeedback(worker_type=worker)
        self.store.workers[worker].record_proposal(
            accepted, confidence, rejection_reason
        )

        # By assertion type
        if assertion_type not in self.store.assertion_feedback:
            self.store.assertion_feedback[assertion_type] = WorkerFeedback(
                worker_type=assertion_type
            )
        self.store.assertion_feedback[assertion_type].record_proposal(
            accepted, confidence, rejection_reason
        )

    def get_worker_stats(self, worker: str) -> WorkerFeedback | None:
        return self.store.workers.get(worker)

    def get_assertion_stats(self, assertion_type: str) -> WorkerFeedback | None:
        return self.store.assertion_feedback.get(assertion_type)

    def get_top_rejection_reasons(self, worker: str, top_n: int = 5) -> list[tuple[str, int]]:
        """Get the most frequent rejection reasons."""
        stats = self.get_worker_stats(worker)
        if not stats:
            return []
        sorted_reasons = sorted(
            stats.rejection_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_reasons[:top_n]
```

---

### 2. Pattern Learner (`pattern_learner.py`)

Analyzes accepted assertions to extract success patterns.

```python
"""
Pattern Learner - Extracts patterns from successful data.
"""

import statistics
from pydantic import BaseModel, Field
from typing import Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class LearnedParameters(BaseModel):
    """Learned parameters for an assertion type."""

    # For geographic clusters
    optimal_radius_km: float | None = None
    radius_std: float | None = None
    optimal_cluster_size: int | None = None
    size_range: tuple[int, int] | None = None

    # For temporal clusters
    optimal_period_overlap: float | None = None

    # For typological clusters
    min_shared_features: int | None = None
    effective_feature_types: list[str] = Field(default_factory=list)

    # For relations
    optimal_distance_range: tuple[float, float] | None = None
    min_similarity_threshold: float | None = None

    # Metadata
    sample_size: int = 0
    last_updated: str | None = None


class PatternStore(BaseModel):
    """Store for learned patterns."""

    version: str = "1.0"
    geographic_clusters: LearnedParameters = Field(default_factory=LearnedParameters)
    chronological_clusters: LearnedParameters = Field(default_factory=LearnedParameters)
    typological_clusters: LearnedParameters = Field(default_factory=LearnedParameters)
    near_to_relations: LearnedParameters = Field(default_factory=LearnedParameters)
    contemporary_relations: LearnedParameters = Field(default_factory=LearnedParameters)
    similar_to_relations: LearnedParameters = Field(default_factory=LearnedParameters)
    thematic_paths: LearnedParameters = Field(default_factory=LearnedParameters)


class PatternLearner:
    """
    Learns optimal patterns from successful data.

    Usage:
        learner = PatternLearner(output_dir)

        # After each run, update with accepted assertions
        learner.learn_from_clusters(
            cluster_type="geographic",
            accepted_clusters=[
                {"radius_km": 45, "size": 4},
                {"radius_km": 52, "size": 5},
            ]
        )

        # Get optimal parameters
        params = learner.get_optimal_params("geographic_clusters")
        print(f"Optimal radius: {params.optimal_radius_km} km")
    """

    DEFAULT_FILENAME = ".learned_patterns.json"
    MIN_SAMPLES_FOR_LEARNING = 5  # Minimum samples to start learning

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.pattern_path = self.output_dir / self.DEFAULT_FILENAME
        self._store: PatternStore | None = None

    @property
    def store(self) -> PatternStore:
        if self._store is None:
            self._store = self._load()
        return self._store

    def _load(self) -> PatternStore:
        if self.pattern_path.exists():
            try:
                with open(self.pattern_path, "r") as f:
                    data = json.load(f)
                return PatternStore(**data)
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
        return PatternStore()

    def save(self) -> None:
        if self._store is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.pattern_path, "w") as f:
            json.dump(self._store.model_dump(), f, indent=2)

    def learn_from_geographic_clusters(
        self,
        accepted_clusters: list[dict[str, Any]],
    ) -> None:
        """Learn from accepted geographic clusters."""
        if len(accepted_clusters) < self.MIN_SAMPLES_FOR_LEARNING:
            return

        radii = [c.get("radius_km") for c in accepted_clusters if c.get("radius_km")]
        sizes = [c.get("size") for c in accepted_clusters if c.get("size")]

        params = self.store.geographic_clusters
        params.sample_size = len(accepted_clusters)

        if radii:
            params.optimal_radius_km = statistics.mean(radii)
            params.radius_std = statistics.stdev(radii) if len(radii) > 1 else 0

        if sizes:
            params.optimal_cluster_size = int(statistics.mean(sizes))
            params.size_range = (min(sizes), max(sizes))

        from datetime import datetime
        params.last_updated = datetime.now().isoformat()

        logger.info(
            "Learned geographic patterns from %d clusters: radius=%.1f±%.1f km, size=%d",
            len(accepted_clusters),
            params.optimal_radius_km or 0,
            params.radius_std or 0,
            params.optimal_cluster_size or 0,
        )

    def learn_from_typological_clusters(
        self,
        accepted_clusters: list[dict[str, Any]],
    ) -> None:
        """Learn from accepted typological clusters."""
        if len(accepted_clusters) < self.MIN_SAMPLES_FOR_LEARNING:
            return

        feature_counts = [
            len(c.get("shared_features", []))
            for c in accepted_clusters
        ]

        # Count which feature types appear most often
        feature_type_counts: dict[str, int] = {}
        for c in accepted_clusters:
            for feature in c.get("shared_features", []):
                # Infer type from feature
                if any(kw in feature.lower() for kw in ["chamber", "corridor", "entrance"]):
                    ftype = "architectural"
                elif any(kw in feature.lower() for kw in ["funerary", "burial", "tomb"]):
                    ftype = "funerary"
                elif any(kw in feature.lower() for kw in ["cult", "ritual", "sacred"]):
                    ftype = "cultic"
                else:
                    ftype = "contextual"
                feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1

        params = self.store.typological_clusters
        params.sample_size = len(accepted_clusters)

        if feature_counts:
            params.min_shared_features = min(feature_counts)

        # Sort feature types by frequency
        params.effective_feature_types = sorted(
            feature_type_counts.keys(),
            key=lambda x: feature_type_counts[x],
            reverse=True,
        )

        from datetime import datetime
        params.last_updated = datetime.now().isoformat()

    def get_optimal_params(self, assertion_type: str) -> LearnedParameters:
        """Get optimal parameters for an assertion type."""
        type_map = {
            "geographic_cluster": self.store.geographic_clusters,
            "chronological_cluster": self.store.chronological_clusters,
            "typological_cluster": self.store.typological_clusters,
            "near_to": self.store.near_to_relations,
            "contemporary_with": self.store.contemporary_relations,
            "similar_to": self.store.similar_to_relations,
            "thematic_path": self.store.thematic_paths,
        }
        return type_map.get(assertion_type, LearnedParameters())

    def has_sufficient_data(self, assertion_type: str) -> bool:
        """Check if there's enough data to use learned parameters."""
        params = self.get_optimal_params(assertion_type)
        return params.sample_size >= self.MIN_SAMPLES_FOR_LEARNING
```

---

### 3. Confidence Adjuster (`confidence_adjuster.py`)

Calibrates LLM confidence scores based on historical data.

```python
"""
Confidence Adjuster - Calibrates LLM confidence scores.
"""

import math
from pydantic import BaseModel, Field
from typing import Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class CalibrationData(BaseModel):
    """Calibration data for an assertion type."""

    # Predicted confidence vs actual acceptance
    predictions: list[tuple[float, bool]] = Field(default_factory=list)

    # Calculated bias (positive = LLM overestimates)
    bias: float = 0.0

    # Correction factor
    correction_factor: float = 1.0

    # Calibrated accuracy
    calibrated_accuracy: float | None = None

    def add_prediction(self, confidence: float, accepted: bool) -> None:
        self.predictions.append((confidence, accepted))

    def recalibrate(self, min_samples: int = 10) -> None:
        """Recalculate bias and correction factor."""
        if len(self.predictions) < min_samples:
            return

        # Calculate mean bias
        # bias = mean(confidence) - mean(acceptance_rate)
        confidences = [p[0] for p in self.predictions]
        acceptances = [1.0 if p[1] else 0.0 for p in self.predictions]

        mean_confidence = sum(confidences) / len(confidences)
        mean_acceptance = sum(acceptances) / len(acceptances)

        self.bias = mean_confidence - mean_acceptance

        # Correction factor (limited between 0.5 and 1.5)
        if mean_confidence > 0:
            self.correction_factor = max(0.5, min(1.5, mean_acceptance / mean_confidence))

        # Calculate calibrated accuracy
        # (how often confidence > 0.5 matches acceptance)
        correct = sum(
            1 for conf, acc in self.predictions
            if (conf > 0.5) == acc
        )
        self.calibrated_accuracy = correct / len(self.predictions)

        logger.info(
            "Recalibrated: bias=%.3f, correction=%.3f, accuracy=%.2f%%",
            self.bias,
            self.correction_factor,
            self.calibrated_accuracy * 100,
        )


class CalibrationStore(BaseModel):
    """Store for calibration data."""

    version: str = "1.0"
    by_worker: dict[str, CalibrationData] = Field(default_factory=dict)
    by_assertion_type: dict[str, CalibrationData] = Field(default_factory=dict)


class ConfidenceAdjuster:
    """
    Calibrates LLM confidence scores based on historical data.

    The problem: LLMs tend to be overly optimistic or pessimistic
    in their confidence estimates. This module corrects the bias.

    Usage:
        adjuster = ConfidenceAdjuster(output_dir)

        # After each validation
        adjuster.record_prediction(
            assertion_type="geographic_cluster",
            predicted_confidence=0.85,
            actually_accepted=True,
        )

        # When calibrating a new confidence
        raw_confidence = 0.80
        calibrated = adjuster.calibrate(
            assertion_type="geographic_cluster",
            confidence=raw_confidence,
        )
        # Might return 0.70 if LLM overestimates by 10%
    """

    DEFAULT_FILENAME = ".confidence_calibration.json"
    MIN_SAMPLES = 10

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.calibration_path = self.output_dir / self.DEFAULT_FILENAME
        self._store: CalibrationStore | None = None

    @property
    def store(self) -> CalibrationStore:
        if self._store is None:
            self._store = self._load()
        return self._store

    def _load(self) -> CalibrationStore:
        if self.calibration_path.exists():
            try:
                with open(self.calibration_path, "r") as f:
                    data = json.load(f)
                return CalibrationStore(**data)
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
        return CalibrationStore()

    def save(self) -> None:
        if self._store is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_path, "w") as f:
            json.dump(self._store.model_dump(), f, indent=2)

    def record_prediction(
        self,
        assertion_type: str,
        predicted_confidence: float,
        actually_accepted: bool,
        worker: str | None = None,
    ) -> None:
        """Record a prediction for future calibration."""
        # By assertion type
        if assertion_type not in self.store.by_assertion_type:
            self.store.by_assertion_type[assertion_type] = CalibrationData()
        self.store.by_assertion_type[assertion_type].add_prediction(
            predicted_confidence, actually_accepted
        )

        # By worker
        if worker:
            if worker not in self.store.by_worker:
                self.store.by_worker[worker] = CalibrationData()
            self.store.by_worker[worker].add_prediction(
                predicted_confidence, actually_accepted
            )

    def recalibrate_all(self) -> None:
        """Recalculate all calibrations."""
        for data in self.store.by_assertion_type.values():
            data.recalibrate(self.MIN_SAMPLES)
        for data in self.store.by_worker.values():
            data.recalibrate(self.MIN_SAMPLES)
        self.save()

    def calibrate(
        self,
        confidence: float,
        assertion_type: str | None = None,
        worker: str | None = None,
    ) -> float:
        """
        Calibrate a raw confidence score.

        Applies correction factor based on historical data.
        If there's not enough data, returns original confidence.
        """
        # Prefer calibration by assertion type
        if assertion_type and assertion_type in self.store.by_assertion_type:
            data = self.store.by_assertion_type[assertion_type]
            if len(data.predictions) >= self.MIN_SAMPLES:
                calibrated = confidence * data.correction_factor
                return max(0.0, min(1.0, calibrated))

        # Fallback to calibration by worker
        if worker and worker in self.store.by_worker:
            data = self.store.by_worker[worker]
            if len(data.predictions) >= self.MIN_SAMPLES:
                calibrated = confidence * data.correction_factor
                return max(0.0, min(1.0, calibrated))

        # No calibration available
        return confidence

    def get_uncertainty(self, assertion_type: str) -> float:
        """
        Calculate calibration uncertainty.

        Lower sample count means higher uncertainty.
        """
        data = self.store.by_assertion_type.get(assertion_type)
        if not data:
            return 1.0  # Maximum uncertainty

        # Uncertainty decreases with sqrt(n)
        return 1.0 / math.sqrt(len(data.predictions) + 1)
```

---

## System Integration

### NodeContext Modifications

```python
class NodeContext:
    def __init__(self, ...):
        # ... existing code ...

        # Option C components
        if output_dir and config.get("learning", {}).get("enabled", False):
            self.feedback_tracker = FeedbackTracker(output_dir)
            self.pattern_learner = PatternLearner(output_dir)
            self.confidence_adjuster = ConfidenceAdjuster(output_dir)
            logger.info("Full Learning System enabled")
        else:
            self.feedback_tracker = None
            self.pattern_learner = None
            self.confidence_adjuster = None
```

### Worker Node Modifications

```python
def geo_analyzer_node(state: dict) -> dict:
    # ... existing analysis ...

    # For each proposal
    for proposal in llm_response.clusters:
        accepted, reason = validate_geographic_cluster(proposal)

        # Option C: record feedback
        if ctx.feedback_tracker:
            ctx.feedback_tracker.record_validation_result(
                worker="geo_analyzer",
                assertion_type="geographic_cluster",
                accepted=accepted,
                confidence=proposal.llm_confidence,
                rejection_reason=reason if not accepted else None,
            )

        # Option C: use calibrated confidence
        if ctx.confidence_adjuster and accepted:
            proposal.confidence_score = ctx.confidence_adjuster.calibrate(
                proposal.llm_confidence,
                assertion_type="geographic_cluster",
            )

    # ... rest of code ...
```

### Adaptive Prompting

```python
class ContextBuilder:
    def build_geo_context(self) -> str:
        context = super().build_geo_context()

        # Add feedback if available
        if self.feedback_tracker:
            stats = self.feedback_tracker.get_worker_stats("geo_analyzer")
            if stats and stats.proposals_total > 10:
                context += f"""

## LEARNING FROM PAST RUNS:
- Your geographic proposals have {stats.acceptance_rate:.0%} acceptance rate
"""
                if stats.acceptance_rate < 0.6:
                    top_reasons = self.feedback_tracker.get_top_rejection_reasons("geo_analyzer", 3)
                    context += "- Common rejection reasons:\n"
                    for reason, count in top_reasons:
                        context += f"  - {reason}: {count} times\n"
                    context += "\nBe more conservative in your proposals.\n"

        return context
```

---

## Configuration

Add to `config.yaml`:

```yaml
learning:
  enabled: false # Disabled by default to avoid overfitting

  feedback:
    track_rejections: true
    min_samples_for_stats: 10

  patterns:
    min_samples_for_learning: 5
    update_interval: "per_run" # or "per_assertion"

  confidence:
    enable_calibration: true
    min_samples: 10
    correction_bounds: [0.5, 1.5]
```

---

## Metrics for the Paper

With Option C you can collect these metrics:

### Table: Learning Effectiveness

| Run | Acceptance Rate | Avg Confidence | Calibrated Accuracy |
| --- | --------------- | -------------- | ------------------- |
| 1   | 45%             | 0.78           | N/A                 |
| 3   | 52%             | 0.75           | 58%                 |
| 5   | 61%             | 0.72           | 67%                 |
| 10  | 68%             | 0.70           | 75%                 |

### Figure: Confidence Calibration

```
Before calibration:
  LLM says 0.80 → Actually accepted 60% of times

After calibration (10 runs):
  LLM says 0.80 → Calibrated to 0.65 → Actually accepted 67% of times
```

### Figure: Parameter Learning

```
Default parameters:
  - max_cluster_radius: 50 km (config)
  - min_shared_features: 2 (config)

Learned parameters (after 10 runs):
  - optimal_radius: 42.3 ± 8.5 km
  - min_shared_features: 3 (learned from failures)
```

---

## Risks and Mitigations

### Risk 1: Overfitting with Small Datasets

**Problem**: With 53 entities, the system might learn dataset-specific patterns instead of general rules.

**Mitigation**:

- Minimum 5-10 runs before using learned parameters
- Limit correction factors (0.5-1.5x)
- Option to disable learning in config

### Risk 2: Temporal Drift

**Problem**: If the dataset changes, learned patterns might become obsolete.

**Mitigation**:

- Temporal decay for old samples
- Periodic calibration reset
- Monitor acceptance rate

### Risk 3: Negative Feedback Loop

**Problem**: If the system becomes too conservative, it might miss valid patterns.

**Mitigation**:

- Don't apply corrections > 50%
- Maintain an "exploration factor"
- Monitor proposal diversity

---

## Implementation Roadmap

1. **Phase 1** (1 day): Implement `FeedbackTracker`
2. **Phase 2** (1 day): Implement `PatternLearner`
3. **Phase 3** (0.5 days): Implement `ConfidenceAdjuster`
4. **Phase 4** (0.5 days): Integrate into worker nodes
5. **Phase 5** (1 day): Testing and validation

**Estimated total: 4 days**

---

## Conclusion

Option C offers a system that improves over time, but requires:

- Larger datasets (>100 entities ideally)
- More runs to see benefits (5-10 minimum)
- Careful monitoring to avoid overfitting

For the TEXT2KG 2026 paper with 53 entities, we recommend:

1. **Use Option B** (already implemented) for submission
2. **Document Option C** as "future work"
3. **Test Option C** on larger datasets in the future

The code in this document is ready to be implemented when you have a larger dataset or want to experiment with learning.
