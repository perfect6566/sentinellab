from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from app.files import SessionFileStore, new_session_id
from app.tools import ProjectTools


class KnownAnswerModel:
    schema = SimpleNamespace(feature_names=("x",))
    threshold = 0.5
    model_version = "known-answer-model"

    @staticmethod
    def normalize_record(record):
        return {"x": float(record["x"])}

    @staticmethod
    def predict_many(records):
        scores = [0.1, 0.8, 0.4, 0.9]
        return [
            SimpleNamespace(
                malware_score=score,
                predicted_label=int(score >= 0.5),
                class_name="malware" if score >= 0.5 else "goodware",
                threshold=0.5,
                model_version="known-answer-model",
            )
            for score in scores[: len(records)]
        ]


def test_evaluation_metrics_match_known_answer() -> None:
    with tempfile.TemporaryDirectory() as directory:
        files = SessionFileStore(Path(directory))
        session_id = new_session_id()
        stored = files.put_csv(
            session_id,
            "known.csv",
            b"x,Label\n1,0\n2,0\n3,1\n4,1\n",
        )
        tools = ProjectTools(model=KnownAnswerModel(), files=files)  # type: ignore[arg-type]

        result = tools.evaluate_labeled(session_id, stored.reference)

    assert result["ok"]
    assert result["data"]["accuracy"] == 0.5
    assert result["data"]["auc"] == 0.75
    assert result["data"]["confusion_matrix"] == {
        "label_order": [0, 1],
        "rows": "true labels",
        "columns": "predicted labels",
        "matrix": [[1, 1], [1, 1]],
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
