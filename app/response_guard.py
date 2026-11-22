"""Deterministic checks for claims in the Agent's final natural-language answer."""

from __future__ import annotations

import json
import math
import re
from typing import Any

NUMBER = re.compile(r"(?<![\w.-])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
CLASSIFICATION_CLAIM = re.compile(
    r"\b(?:is|was)\s+(?:(?:classified|predicted|labeled|labelled)\s+as\s+)?"
    r"(?:malware|goodware)\b|"
    r"\b(?:classified|predicted|labeled|labelled)\s+as\s+(?:malware|goodware)\b"
)
# The score is uncalibrated, so it must not be presented as a confidence level.
# Negated mentions ("not a confidence measure") state that limitation correctly and
# are allowed; an actual confidence claim is not.
CONFIDENCE_CLAIM = re.compile(
    r"\b(?:is|are|was|were|be|being|been|feel|feels|looks?|seems?|appears?|remain|remains)\s+"
    r"(?:very|highly|extremely|quite|fairly|reasonably|perfectly|totally|absolutely|more|less)?\s*"
    r"confident\b"
    r"|\b(?:high|strong|great|good|full|complete|total|absolute)\s+confidence\b"
    r"|\bconfidence\s+(?:level|of|score|is|was|remains?|value)\b"
    r"|\bwith\s+confidence\b"
    r"|\bconfidently\b"
    r"|\b(?:i|we)\s+am\s+(?:very\s+|highly\s+|fairly\s+|quite\s+|fully\s+)?confident\b",
    re.IGNORECASE,
)
# A negated or hypothetical mention states that the limitation applies and is allowed.
CONFIDENCE_NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|can't|without|nor|neither|nothing|none)\b[^.!?\n]{0,60}"
    r"\bconfidence\b"
    r"|\bconfidence\b[^.!?\n]{0,60}"
    r"\b(?:would be|could be|is not|are not|isn't|aren't|would be misleading|is misleading)\b"
    r"|\b(?:misleading|meaningless|not applicable|n/a|unsupported)\b[^.!?\n]{0,40}\bconfidence\b",
    re.IGNORECASE,
)

# Words that make a nearby number a quantitative claim about the model output.
METRIC_KEYWORDS = (
    "accuracy",
    "auc",
    "area under",
    "score",
    "threshold",
    "probability",
    "confidence",
    "precision",
    "recall",
    "f1",
    "false negative",
    "false positive",
    "tn",
    "fp",
    "fn",
    "tp",
    "true negative",
    "true positive",
    "sample",
    "row",
    "count",
    "number of",
    "percent",
    "percentage",
    "%",
)

# Structural or ordinal numbers are presentation, not quantitative claims.
# "Step 2 - Accuracy gate" and "1. Evaluate the file" must not be read as
# metric assertions, so they are exempt from numeric grounding.
ORDINAL_CONTEXT = re.compile(
    r"(?:\b(?:step|phase|stage|part|section|chapter|item|note|point)\s*"
    r"\d+|\b\d+(?:st|nd|rd|th)\b|^\s*\d+[.)]\s|\bversion\s*\d+)",
    re.IGNORECASE,
)
SECTION_HEADING = re.compile(r"^\s*#{1,6}\s*.*\d+.*$", re.MULTILINE)


def _numbers(value: Any) -> set[float]:
    found: set[float] = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        number = float(value)
        found.add(round(number, 12))
        if 0.0 <= number <= 1.0:
            found.add(round(number * 100.0, 12))
        return found
    if isinstance(value, dict):
        for item in value.values():
            found.update(_numbers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_numbers(item))
    return found


def _number_values(text: str) -> set[float]:
    """Return the numeric values stated in a piece of text."""
    values: set[float] = set()
    for token in NUMBER.findall(text):
        is_percent = token.endswith("%")
        value = float(token.rstrip("%").replace(",", ""))
        values.add(round(value if is_percent else value, 12))
    return values


def _strip_structural(answer: str) -> str:
    """Remove references, headings, and ordinal markers that carry no metric claim."""
    without_uuids = UUID.sub(" ", answer)
    without_headings = SECTION_HEADING.sub("", without_uuids)
    return ORDINAL_CONTEXT.sub(" ", without_headings)


def _metric_claims(answer: str) -> set[float]:
    """Return only the numbers that answer to a metric keyword.

    A number is treated as a quantitative claim when a metric word appears in
    the same sentence. Numbers in other sentences are presentation detail such
    as step numbers or a restated threshold, and are not metric assertions.
    """
    lowered = answer.lower()
    claims: set[float] = set()
    for sentence in re.split(r"(?<=[.!?\n])\s+", lowered):
        if not any(keyword in sentence for keyword in METRIC_KEYWORDS):
            continue
        claims.update(_number_values(sentence))
    return claims


def validate_final_answer(
    answer: str,
    grounded_results: list[dict[str, Any]],
    *,
    trusted_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Return an error reason when an answer exceeds the available tool evidence."""
    lowered = answer.lower()
    if not grounded_results and CLASSIFICATION_CLAIM.search(lowered):
        return "The model attempted to classify without a tool result"
    if CONFIDENCE_CLAIM.search(lowered) and not CONFIDENCE_NEGATION.search(lowered):
        return "The model described an uncalibrated score as confidence"
    explanation_markers = ("feature caused", "because of the feature", "due to the feature")
    if any(marker in lowered for marker in explanation_markers):
        return "The model attempted a feature-level explanation without an explanation tool"
    allowed = _numbers(grounded_results)
    if trusted_metadata:
        allowed.update(_numbers(trusted_metadata))
    unsupported = {
        claim
        for claim in _metric_claims(_strip_structural(answer))
        if not any(math.isclose(claim, value, rel_tol=1e-5, abs_tol=5e-6) for value in allowed)
    }
    if (grounded_results or trusted_metadata) and unsupported:
        return (
            "The answer contains numeric claims not present in trusted context or tool results: "
            + json.dumps(sorted(unsupported))
        )
    return None
