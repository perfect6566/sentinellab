"""Frontend must keep the labeled evaluation as the follow-up context.

The conditional task evaluates first and predicts second, so the last
result in the batch is the prediction. A follow-up question about
false negatives needs the evaluation result, which carries the confusion
matrix. This test guards the selection rule in app/static/app.js.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[2] / "app" / "static" / "app.js"


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_followup_context_selects_the_labeled_evaluation() -> None:
    source = _source()
    assert 'tool === "evaluate_labeled"' in source, (
        "the follow-up context must select the labeled evaluation result"
    )


def test_followup_context_falls_back_to_last_result_without_evaluation() -> None:
    """The rule must be conditional, not an unconditional last-result pick."""
    source = _source()
    # Find the block that assigns state.latestResultId from the result batch.
    block = re.search(
        r"if \(resultIds\.length\) \{(.*?)\n    \}",
        source,
        re.S,
    )
    assert block, "state.latestResultId must be assigned inside the result batch block"
    body = block.group(1)
    assert "evaluate_labeled" in body, "the block must look for an evaluation result"
    assert "resultIds.at(-1)" in body, "the block must keep the last-result fallback"
    # The evaluation branch must actually be reachable.
    assert re.search(r"if \((evaluationIndex|evaluationIndex\s*!==\s*-1)\)", body), (
        "the evaluation branch must be guarded by a real condition"
    )
    assert "if (false)" not in body, "the evaluation branch must not be disabled"
