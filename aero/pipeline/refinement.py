"""
Hypothesis refinement module for Aero Agent.

Refines hypotheses based on validation results and detected patterns.
Uses deterministic rules (no LLM) for testable, reproducible behavior.
"""

import logging
from typing import Any, Dict, List, Optional
from copy import deepcopy

from aero.pipeline.hypothesis import Hypothesis
from aero.pipeline.validator import SimulationValidationResult
from aero.pipeline.patterns import ScientificPattern

logger = logging.getLogger(__name__)


# =============================================================================
# Refinement Configuration
# =============================================================================

# Confidence adjustment factors
CONFIDENCE_ADJUSTMENTS = {
    "validation_strong": 0.15,    # Validation score >= 0.8
    "validation_good": 0.10,      # Validation score >= 0.6
    "validation_weak": -0.05,     # Validation score >= 0.4
    "validation_poor": -0.15,     # Validation score < 0.4
    "validation_failed": -0.25,   # Validation had issues

    "pattern_match": 0.10,        # Patterns match hypothesis domain
    "pattern_mismatch": -0.05,    # Patterns don't match
    "pattern_converged": 0.15,    # Convergence/steady-state detected

    "iteration_bonus": 0.02,      # Small bonus for surviving iterations
}

# Text mutation templates
MUTATION_TEMPLATES = {
    "strengthen": [
        "The evidence strongly suggests that {original}",
        "Results confirm that {original}",
        "Simulation validates that {original}",
    ],
    "weaken": [
        "There is limited evidence that {original}",
        "Results partially support that {original}",
        "With some uncertainty, {original}",
    ],
    "modify_converged": [
        "The converged solution shows {original}",
        "At steady state, {original}",
    ],
    "modify_pattern": [
        "The observed {pattern} pattern indicates {original}",
        "Pattern analysis reveals {original}",
    ],
}


# =============================================================================
# Main Refinement Function
# =============================================================================


def refine_hypotheses(
    old_hypotheses: List[Hypothesis],
    validation_data: List[SimulationValidationResult],
    patterns: List[ScientificPattern],
    drop_threshold: float = 0.2,
    boost_threshold: float = 0.7,
) -> List[Hypothesis]:
    """
    Refine hypotheses based on validation results and detected patterns.

    Refinement rules:
    1. Increase confidence when validation is strong
    2. Decrease confidence when validation fails
    3. Adjust based on detected patterns
    4. Drop hypotheses below threshold
    5. Mutate text with simple rules

    Args:
        old_hypotheses: Previous hypotheses to refine
        validation_data: Validation results for each hypothesis
        patterns: Detected scientific patterns
        drop_threshold: Drop hypotheses below this confidence
        boost_threshold: Strongly boost hypotheses above this validation score

    Returns:
        List of refined Hypothesis objects
    """
    logger.info(f"Refining {len(old_hypotheses)} hypotheses")

    if not old_hypotheses:
        logger.warning("No hypotheses to refine")
        return []

    refined = []

    for i, hypothesis in enumerate(old_hypotheses):
        # Get corresponding validation result
        validation = None
        if i < len(validation_data):
            validation = validation_data[i]

        # Refine the hypothesis
        new_hypothesis = _refine_single_hypothesis(
            hypothesis=hypothesis,
            validation=validation,
            patterns=patterns,
            boost_threshold=boost_threshold,
        )

        # Check if hypothesis should be dropped
        if new_hypothesis.confidence >= drop_threshold:
            refined.append(new_hypothesis)
            logger.debug(
                f"Refined hypothesis {hypothesis.id}: "
                f"confidence {hypothesis.confidence:.2f} -> {new_hypothesis.confidence:.2f}"
            )
        else:
            logger.info(
                f"Dropping hypothesis {hypothesis.id}: "
                f"confidence {new_hypothesis.confidence:.2f} < {drop_threshold}"
            )

    # Sort by confidence (highest first)
    refined.sort(key=lambda h: h.confidence, reverse=True)

    logger.info(f"Refinement complete: {len(refined)} hypotheses remaining")
    return refined


def _refine_single_hypothesis(
    hypothesis: Hypothesis,
    validation: Optional[SimulationValidationResult],
    patterns: List[ScientificPattern],
    boost_threshold: float,
) -> Hypothesis:
    """Refine a single hypothesis."""
    # Create a copy to modify
    new_hypothesis = Hypothesis(
        id=hypothesis.id,
        text=hypothesis.text,
        confidence=hypothesis.confidence,
        metadata=deepcopy(hypothesis.metadata),
        parent_id=hypothesis.parent_id or hypothesis.id,
    )

    # Track adjustments for metadata
    adjustments = []

    # Adjust based on validation
    if validation is not None:
        confidence_delta = _calculate_validation_adjustment(validation, boost_threshold)
        new_hypothesis.confidence += confidence_delta
        adjustments.append(f"validation:{confidence_delta:+.2f}")

        # Mutate text based on validation strength
        new_hypothesis.text = _mutate_text_validation(
            hypothesis.text,
            validation.score,
        )

    # Adjust based on patterns
    pattern_delta = _calculate_pattern_adjustment(hypothesis, patterns)
    new_hypothesis.confidence += pattern_delta
    if pattern_delta != 0:
        adjustments.append(f"patterns:{pattern_delta:+.2f}")

    # Mutate text based on patterns
    if patterns:
        new_hypothesis.text = _mutate_text_patterns(
            new_hypothesis.text,
            patterns,
        )

    # Small iteration survival bonus
    iteration_bonus = CONFIDENCE_ADJUSTMENTS["iteration_bonus"]
    new_hypothesis.confidence += iteration_bonus
    adjustments.append(f"iteration:{iteration_bonus:+.2f}")

    # Clamp confidence to [0, 1]
    new_hypothesis.confidence = max(0.0, min(1.0, new_hypothesis.confidence))

    # Update metadata
    new_hypothesis.metadata["refined"] = True
    new_hypothesis.metadata["adjustments"] = adjustments
    new_hypothesis.metadata["validation_score"] = validation.score if validation else None
    new_hypothesis.metadata["pattern_count"] = len(patterns)

    return new_hypothesis


def _calculate_validation_adjustment(
    validation: SimulationValidationResult,
    boost_threshold: float,
) -> float:
    """Calculate confidence adjustment from validation results."""
    score = validation.score

    if not validation.valid:
        return CONFIDENCE_ADJUSTMENTS["validation_failed"]

    if score >= boost_threshold:
        return CONFIDENCE_ADJUSTMENTS["validation_strong"]
    elif score >= 0.6:
        return CONFIDENCE_ADJUSTMENTS["validation_good"]
    elif score >= 0.4:
        return CONFIDENCE_ADJUSTMENTS["validation_weak"]
    else:
        return CONFIDENCE_ADJUSTMENTS["validation_poor"]


def _calculate_pattern_adjustment(
    hypothesis: Hypothesis,
    patterns: List[ScientificPattern],
) -> float:
    """Calculate confidence adjustment from detected patterns."""
    if not patterns:
        return 0.0

    adjustment = 0.0
    domain = hypothesis.metadata.get("domain", "")
    text_lower = hypothesis.text.lower()

    # Check for convergence patterns
    for pattern in patterns:
        if pattern.type in ["convergence", "steady_state"]:
            adjustment += CONFIDENCE_ADJUSTMENTS["pattern_converged"]
            break

    # Check for domain-relevant patterns
    domain_patterns = {
        "heat_transfer": ["diffusion_profile", "monotonic_decreasing", "steady_state"],
        "fluid_dynamics": ["symmetry", "x_symmetry", "y_symmetry"],
        "aerodynamics": ["symmetry", "linear_gradient"],
    }

    relevant_types = domain_patterns.get(domain, [])
    for pattern in patterns:
        if pattern.type in relevant_types:
            adjustment += CONFIDENCE_ADJUSTMENTS["pattern_match"]
        elif pattern.type in ["oscillations", "shock_gradient"]:
            # These might indicate issues
            if "stable" in text_lower or "smooth" in text_lower:
                adjustment += CONFIDENCE_ADJUSTMENTS["pattern_mismatch"]

    return adjustment


def _mutate_text_validation(text: str, validation_score: float) -> str:
    """Mutate hypothesis text based on validation strength."""
    if validation_score >= 0.8:
        templates = MUTATION_TEMPLATES["strengthen"]
    elif validation_score < 0.4:
        templates = MUTATION_TEMPLATES["weaken"]
    else:
        return text  # No mutation for moderate scores

    # Use first template (deterministic)
    template = templates[0]
    return template.format(original=text.lower())


def _mutate_text_patterns(text: str, patterns: List[ScientificPattern]) -> str:
    """Mutate hypothesis text based on detected patterns."""
    if not patterns:
        return text

    # Find most significant pattern
    best_pattern = max(patterns, key=lambda p: p.confidence)

    # Check for convergence
    if best_pattern.type in ["convergence", "steady_state"]:
        templates = MUTATION_TEMPLATES["modify_converged"]
        return templates[0].format(original=text.lower())

    # Check for other significant patterns
    if best_pattern.confidence >= 0.7:
        templates = MUTATION_TEMPLATES["modify_pattern"]
        return templates[0].format(
            pattern=best_pattern.type.replace("_", " "),
            original=text.lower(),
        )

    return text


# =============================================================================
# Utility Functions
# =============================================================================


def merge_hypotheses(
    hypotheses: List[Hypothesis],
    similarity_threshold: float = 0.8,
) -> List[Hypothesis]:
    """
    Merge similar hypotheses to reduce redundancy.

    Args:
        hypotheses: List of hypotheses
        similarity_threshold: Merge if similarity exceeds this

    Returns:
        Merged list of hypotheses
    """
    if len(hypotheses) <= 1:
        return hypotheses

    merged = []
    used = set()

    for i, h1 in enumerate(hypotheses):
        if i in used:
            continue

        # Find similar hypotheses
        similar_group = [h1]
        for j, h2 in enumerate(hypotheses[i+1:], i+1):
            if j in used:
                continue

            similarity = _text_similarity(h1.text, h2.text)
            if similarity >= similarity_threshold:
                similar_group.append(h2)
                used.add(j)

        # Merge the group
        if len(similar_group) == 1:
            merged.append(h1)
        else:
            merged_hypothesis = _merge_hypothesis_group(similar_group)
            merged.append(merged_hypothesis)

        used.add(i)

    return merged


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (word overlap)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _merge_hypothesis_group(group: List[Hypothesis]) -> Hypothesis:
    """Merge a group of similar hypotheses into one."""
    # Use highest confidence hypothesis as base
    best = max(group, key=lambda h: h.confidence)

    # Average confidence with slight boost for agreement
    avg_confidence = sum(h.confidence for h in group) / len(group)
    boosted_confidence = min(1.0, avg_confidence + 0.05 * (len(group) - 1))

    return Hypothesis(
        id=best.id,
        text=best.text,
        confidence=boosted_confidence,
        metadata={
            **best.metadata,
            "merged_from": [h.id for h in group],
            "merge_count": len(group),
        },
        parent_id=best.parent_id,
    )


def rank_by_validation(
    hypotheses: List[Hypothesis],
    validation_results: List[SimulationValidationResult],
) -> List[Hypothesis]:
    """
    Rank hypotheses by their validation scores.

    Args:
        hypotheses: Hypotheses to rank
        validation_results: Corresponding validation results

    Returns:
        Sorted list (best validation first)
    """
    if not validation_results:
        return hypotheses

    # Pair hypotheses with scores
    pairs = []
    for i, h in enumerate(hypotheses):
        score = validation_results[i].score if i < len(validation_results) else 0.0
        pairs.append((h, score))

    # Sort by validation score
    pairs.sort(key=lambda p: p[1], reverse=True)

    return [h for h, _ in pairs]


def generate_refinement_summary(
    old_hypotheses: List[Hypothesis],
    new_hypotheses: List[Hypothesis],
    validation_results: List[SimulationValidationResult],
) -> Dict[str, Any]:
    """
    Generate a summary of the refinement process.

    Args:
        old_hypotheses: Original hypotheses
        new_hypotheses: Refined hypotheses
        validation_results: Validation results

    Returns:
        Summary dictionary
    """
    dropped_count = len(old_hypotheses) - len(new_hypotheses)
    dropped_ids = [
        h.id for h in old_hypotheses
        if h.id not in {n.id for n in new_hypotheses}
    ]

    confidence_changes = []
    for new_h in new_hypotheses:
        old_h = next((h for h in old_hypotheses if h.id == new_h.id), None)
        if old_h:
            confidence_changes.append({
                "id": new_h.id,
                "old": old_h.confidence,
                "new": new_h.confidence,
                "delta": new_h.confidence - old_h.confidence,
            })

    avg_validation = (
        sum(v.score for v in validation_results) / len(validation_results)
        if validation_results else 0.0
    )

    return {
        "original_count": len(old_hypotheses),
        "refined_count": len(new_hypotheses),
        "dropped_count": dropped_count,
        "dropped_ids": dropped_ids,
        "confidence_changes": confidence_changes,
        "average_validation_score": avg_validation,
        "best_hypothesis": new_hypotheses[0].to_dict() if new_hypotheses else None,
    }
