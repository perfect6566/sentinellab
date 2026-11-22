"""The three model-backed tools available to the AI Agent."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from app.files import FileStoreError, SessionFileStore
from app.inference import ModelService, get_model_service
from ml.contracts import ContractError

ALLOWED_AUXILIARY_COLUMNS = {"Label", "SHA1", "FirstSeenDate", "row_id"}


class ProjectTools:
    def __init__(
        self,
        model: ModelService | None = None,
        files: SessionFileStore | None = None,
        max_rows: int = 10_000,
    ) -> None:
        self.model = model or get_model_service()
        self.files = files or SessionFileStore()
        self.max_rows = max_rows

    @staticmethod
    def _failure(tool: str, code: str, message: str, **details: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool,
            "error": {"code": code, "message": message, "details": details},
            "activity": {"tool": tool, "status": "failed", "error_code": code},
        }

    def predict_single(self, record: Mapping[str, Any]) -> dict[str, Any]:
        tool = "predict_single"
        try:
            prediction = self.model.predict_one(record)
        except (ContractError, ValueError) as exc:
            return self._failure(tool, "invalid_record", str(exc), supplied_field_count=len(record))
        feature_count = len(record)
        return {
            "ok": True,
            "tool": tool,
            # The validated feature count belongs in the result data, not only in
            # the activity record: the data is what a follow-up question receives
            # as its trusted context, so a count kept only in the activity could
            # not be grounded and a correct answer would be rejected.
            "data": {**asdict(prediction), "validated_feature_count": feature_count},
            "activity": {
                "tool": tool,
                "status": "succeeded",
                "validated_feature_count": feature_count,
                "model_version": prediction.model_version,
                "predicted_label": prediction.predicted_label,
            },
        }

    def _read_rows(self, session_id: str, reference: str) -> tuple[list[str], list[dict[str, str]]]:
        stored = self.files.resolve(session_id, reference)
        text = stored.path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        header = [name.strip() for name in reader.fieldnames]
        if len(header) != len(set(header)) or any(not name for name in header):
            raise ValueError("CSV header contains empty or duplicate names")
        rows = []
        for index, row in enumerate(reader, start=1):
            if index > self.max_rows:
                raise ValueError(f"CSV exceeds {self.max_rows} row limit")
            if None in row:
                raise ValueError(f"CSV row {index} has more values than the header")
            rows.append({str(key).strip(): (value or "").strip() for key, value in row.items()})
        if not rows:
            raise ValueError("CSV contains no data rows")
        return header, rows

    def _model_record(self, row: Mapping[str, str]) -> dict[str, str]:
        return {name: row[name] for name in self.model.schema.feature_names if name in row}

    @staticmethod
    def _stable_ids(rows: list[dict[str, str]]) -> list[str]:
        used: Counter[str] = Counter()
        result = []
        for index, row in enumerate(rows, start=1):
            base = row.get("row_id") or row.get("SHA1") or f"row-{index}"
            used[base] += 1
            result.append(base if used[base] == 1 else f"{base}#{used[base]}")
        return result

    def predict_batch(self, session_id: str, file_reference: str) -> dict[str, Any]:
        tool = "predict_batch"
        try:
            header, rows = self._read_rows(session_id, file_reference)
        except (FileStoreError, OSError, UnicodeError, ValueError) as exc:
            return self._failure(tool, "invalid_file", str(exc))
        required = set(self.model.schema.feature_names)
        missing_header = sorted(required - set(header))
        unknown = sorted(set(header) - required - ALLOWED_AUXILIARY_COLUMNS)
        if missing_header or unknown:
            return self._failure(
                tool,
                "invalid_schema",
                "CSV schema does not match production input",
                missing_features=missing_header,
                unknown_columns=unknown,
            )

        stable_ids = self._stable_ids(rows)
        results: list[dict[str, Any]] = []
        valid_records = []
        valid_positions = []
        for index, (row_id, row) in enumerate(zip(stable_ids, rows, strict=True), start=1):
            try:
                normalized = self.model.normalize_record(self._model_record(row))
                valid_records.append(normalized)
                valid_positions.append(len(results))
                results.append({"input_row": index, "row_id": row_id, "status": "pending"})
            except (ContractError, ValueError) as exc:
                results.append(
                    {"input_row": index, "row_id": row_id, "status": "invalid", "error": str(exc)}
                )
        predictions = self.model.predict_many(valid_records)
        for position, prediction in zip(valid_positions, predictions, strict=True):
            results[position].update(
                {
                    "status": "predicted",
                    "predicted_label": prediction.predicted_label,
                    "class_name": prediction.class_name,
                    "malware_score": prediction.malware_score,
                    "threshold": prediction.threshold,
                    "model_version": prediction.model_version,
                }
            )

        output = io.StringIO()
        fields = [
            "input_row",
            "row_id",
            "status",
            "predicted_label",
            "class_name",
            "malware_score",
            "threshold",
            "model_version",
            "error",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        stored_result = self.files.put_csv(
            session_id, "batch-predictions.csv", output.getvalue().encode(), kind="batch_result"
        )
        predicted = [item for item in results if item["status"] == "predicted"]
        counts = Counter(item["predicted_label"] for item in predicted)
        summary = {
            "input_rows": len(rows),
            "valid_rows": len(predicted),
            "invalid_rows": len(rows) - len(predicted),
            "goodware_count": counts[0],
            "malware_count": counts[1],
        }
        ok = bool(predicted)
        response = {
            "ok": ok,
            "tool": tool,
            "data": {
                "summary": summary,
                "rows": results[:100],
                "rows_truncated": len(results) > 100,
                "result_file_reference": stored_result.reference,
                "model_version": self.model.model_version,
            },
            "activity": {
                "tool": tool,
                "status": "succeeded" if ok else "failed",
                **summary,
                "model_version": self.model.model_version,
            },
        }
        if not ok:
            response["error"] = {"code": "no_valid_rows", "message": "No rows were valid"}
        return response

    def evaluate_labeled(self, session_id: str, file_reference: str) -> dict[str, Any]:
        tool = "evaluate_labeled"
        try:
            stored_file = self.files.resolve(session_id, file_reference)
            header, rows = self._read_rows(session_id, file_reference)
        except (FileStoreError, OSError, UnicodeError, ValueError) as exc:
            return self._failure(tool, "invalid_file", str(exc))
        if "Label" not in header:
            return self._failure(tool, "missing_labels", "CSV must contain a Label column")
        required = set(self.model.schema.feature_names)
        missing_header = sorted(required - set(header))
        unknown = sorted(set(header) - required - ALLOWED_AUXILIARY_COLUMNS)
        if missing_header or unknown:
            return self._failure(
                tool,
                "invalid_schema",
                "CSV schema does not match labeled evaluation input",
                missing_features=missing_header,
                unknown_columns=unknown,
            )

        records = []
        labels = []
        errors = []
        for index, row in enumerate(rows, start=1):
            try:
                label = int(row["Label"])
                if label not in {0, 1}:
                    raise ValueError("Label must be 0 or 1")
                records.append(self.model.normalize_record(self._model_record(row)))
                labels.append(label)
            except (ContractError, ValueError) as exc:
                errors.append({"input_row": index, "error": str(exc)})
        if errors:
            return self._failure(
                tool,
                "invalid_rows",
                "Evaluation aborted because one or more rows are invalid",
                invalid_row_count=len(errors),
                sample_errors=errors[:20],
            )

        predictions = self.model.predict_many(records)
        scores = np.asarray([item.malware_score for item in predictions])
        predicted_labels = np.asarray([item.predicted_label for item in predictions])
        truth = np.asarray(labels)
        matrix = confusion_matrix(truth, predicted_labels, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        unique = sorted(set(labels))
        auc = float(roc_auc_score(truth, scores)) if len(unique) == 2 else None
        auc_reason = (
            None if auc is not None else "AUC is undefined because only one true class is present"
        )
        built_in_fixture = (
            re.fullmatch(
                r"labeled-sample(?: \(\d+\))?\.csv",
                stored_file.original_name,
                flags=re.IGNORECASE,
            )
            is not None
        )
        data = {
            "file_name": stored_file.original_name,
            "file_reference": stored_file.reference,
            "sample_count": len(labels),
            "true_class_counts": {str(label): labels.count(label) for label in (0, 1)},
            "class_mapping": {"0": "goodware", "1": "malware"},
            "auc": auc,
            "auc_reason": auc_reason,
            "accuracy": float(accuracy_score(truth, predicted_labels)),
            "confusion_matrix": {
                "label_order": [0, 1],
                "rows": "true labels",
                "columns": "predicted labels",
                "matrix": matrix.tolist(),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
            "threshold": self.model.threshold,
            "model_version": self.model.model_version,
            "interpretation_note": (
                "This is the built-in workflow fixture, sampled from the development partition "
                "used to fit the production model; its metrics are not a valid estimate of "
                "performance on unseen files."
                if built_in_fixture
                else "Metrics describe only this uploaded CSV. Independence from training data "
                "and representativeness of future files are not verified by the model."
            ),
        }
        return {
            "ok": True,
            "tool": tool,
            "data": data,
            "activity": {
                "tool": tool,
                "status": "succeeded",
                "sample_count": len(labels),
                "auc": auc,
                "accuracy": data["accuracy"],
                "model_version": self.model.model_version,
            },
        }
