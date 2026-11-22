# Dataset Audit and Modeling-Schema Decision

Audited: 2026-10-28  
Source file: `data/raw/brazilian-malware.csv`  
Machine-readable report: `reports/data-validation.json`

## Provenance and safety

The user manually downloaded the public `fabriciojoc/brazilian-malware-dataset` repository archive. The archive contains CSV data and a nested ZIP. Only the nested structured `brazilian-malware.csv` was extracted into this project. No executable malware was obtained or run.

The repository README asks users to cite the associated 2018 IEEE Security & Privacy paper. The downloaded archive contains no explicit LICENSE file, so this project must not assume redistribution rights. Raw data remains ignored by Git; the repository should provide retrieval instructions and hashes instead.

The downloaded `master` snapshot does not embed a verifiable commit SHA in its filename. Retrieval date, byte size and content hashes are therefore the authoritative local version identifiers. See `data/metadata/source.json`.

## Assignment-schema verification

| Check | Observed | Decision |
|---|---:|---|
| Records | 50,181 | Matches the assignment's approximate 50,000 records |
| Total columns | 28 | 27 non-label attributes plus one target |
| Target | `Label` (12th column) | Matches the prompt's Column L description |
| Goodware | 21,116 (`42.08%`) | Mapping confirmed as `0 = goodware` |
| Malware | 29,065 (`57.92%`) | Mapping confirmed as `1 = malware` |
| Missing target values | 0 | Acceptable |
| Unexpected target values | 0 | Acceptable |
| Raw CSV SHA-256 | `d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242` | Recorded for reproducibility |

The raw schema therefore matches the assignment description.

## Feature roles

The source has 27 non-label attributes. They are not all defensible predictive inputs:

- `SHA1` is a file identity/hash, not a behavioral PE feature. It is excluded from the model and retained as the stable grouping key.
- `FirstSeenDate` is collection metadata. Its observed ranges differ sharply by label (`goodware: 1970-01-01 through 2100-02-26`; `malware: 2013-01-02 through 2019-07-08`), making it a material leakage risk. It is excluded from the initial model.
- The other 25 columns are initial model candidates. `Identify`, `ImportedDlls` and `ImportedSymbols` need fold-local categorical/text handling.
- `Magic`, `PE_TYPE` and `SizeOfOptionalHeader` are constant in this raw snapshot. Their removal, if used, must occur inside the training pipeline so evaluation does not learn transformations from held-out data.

The exact roles are machine-readable in `data/metadata/feature-roles.json`.

## Missing values

`Identify` has 14,223 empty values: 9,364 goodware and 4,859 malware. All other columns have no missing values according to the raw string-level audit. Missing-category handling for `Identify` must be fitted/applied consistently inside the pipeline.

## Duplicate and related-record risk

| Finding | Count |
|---|---:|
| Exact duplicate extra rows | 2,480 |
| Unique SHA1 groups | 43,411 |
| SHA1 groups with more than one row | 3,542 |
| Extra rows beyond one row per SHA1 | 6,770 |
| SHA1 groups containing both labels | 18 groups / 54 rows |
| Maximum rows under one SHA1 | 197 |

This is a substantial leakage risk. A simple row-wise stratified split could put observations of the same file on both sides.

Initial handling decision for the split task:

1. Remove all 18 conflicting-label SHA1 groups (54 rows: 18 goodware, 36 malware) and record their IDs only in a generated audit artifact.
2. Remove exact duplicate rows deterministically after conflict removal. This leaves 47,653 rows: 21,098 goodware and 26,555 malware.
3. Keep non-identical observations sharing a SHA1, but ensure every SHA1 group stays in exactly one partition and one CV fold.
4. Use SHA1-aware stratification for the final 80/20 split and development-set 10-fold evaluation.
5. Report row-level metrics and the number of unique SHA1 groups so repeated observations remain transparent.

These rules prevent identity leakage without pretending ambiguous labels are trustworthy. Their implementation and resulting partition balance must be verified in S2 before model training.

## Differences and limitations

- The source matches the stated record/attribute/label structure.
- The defensible initial model uses 25 of the 27 source attributes because SHA1 is an identifier and FirstSeenDate is collection-time leakage risk. This is a documented feature-selection decision, not a silent schema change.
- The repository archive does not include an explicit software/data license or immutable commit identifier. Citation information is present, but redistribution permission remains unconfirmed.
- Full statistical exploration and learned preprocessing remain development-set-only work after the 20% test partition is sealed.

## Verification commands

```text
shasum -a 256 data/raw/brazilian-malware.csv
python -m ml.data_audit data/raw/brazilian-malware.csv \
  --expected-feature-count 27 \
  --output reports/data-validation.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

All three checks were run. The schema audit passed and seven data-audit unit tests passed.
