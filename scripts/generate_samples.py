"""Generate small, development-only grader samples from the sealed schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ml.dataset import load_partition


def main() -> int:
    data = load_partition("development")
    output = Path("samples")
    output.mkdir(parents=True, exist_ok=True)
    text_length = (
        data.features["ImportedDlls"].str.len() + data.features["ImportedSymbols"].str.len()
    )
    single_index = int(text_length.idxmin())
    single = data.features.iloc[single_index].to_dict()
    (output / "single-record.json").write_text(
        json.dumps({"record": single}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    batch_indices = list(range(5))
    with (output / "batch-sample.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["row_id", *data.features.columns]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in batch_indices:
            writer.writerow(
                {"row_id": f"sample-{index + 1}", **data.features.iloc[index].to_dict()}
            )

    goodware = [index for index, label in enumerate(data.labels) if label == 0][:10]
    malware = [index for index, label in enumerate(data.labels) if label == 1][:10]
    with (output / "labeled-sample.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["row_id", *data.features.columns, "Label"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ordinal, index in enumerate(goodware + malware, start=1):
            writer.writerow(
                {
                    "row_id": f"labeled-{ordinal}",
                    **data.features.iloc[index].to_dict(),
                    "Label": int(data.labels[index]),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
