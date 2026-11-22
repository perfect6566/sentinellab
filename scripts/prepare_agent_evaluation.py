"""Generate clearly labeled synthetic fixtures for Agent flow evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "samples/labeled-sample.csv"
OUTPUT = ROOT / "samples/evaluation"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows or "Label" not in fieldnames:
        raise RuntimeError("The labeled sample is empty or missing Label")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    flipped = [dict(row, Label="0" if row["Label"] == "1" else "1") for row in rows]
    write_rows(OUTPUT / "labels-flipped.csv", fieldnames, flipped)

    without_label = [name for name in fieldnames if name != "Label"]
    write_rows(OUTPUT / "missing-label.csv", without_label, rows[:5])

    first_class = rows[0]["Label"]
    single_class = [row for row in rows if row["Label"] == first_class][:5]
    write_rows(OUTPUT / "single-class.csv", fieldnames, single_class)

    invalid = [dict(rows[0])]
    invalid[0]["Entropy"] = "not-a-number"
    write_rows(OUTPUT / "invalid-feature.csv", fieldnames, invalid)

    readme = OUTPUT / "README.md"
    readme.write_text(
        "# Agent evaluation fixtures\n\n"
        "These files are synthetic flow-test derivatives of `samples/labeled-sample.csv`. "
        "They test error handling and conditional routing; they are not evidence of production "
        "model quality. `labels-flipped.csv` deliberately reverses labels.\n",
        encoding="utf-8",
    )
    print(f"Prepared evaluation fixtures in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
