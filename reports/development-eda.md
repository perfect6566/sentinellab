# Development-Partition Data Quality Report

Generated: 2026-10-28  
Machine-readable profile: `reports/development-eda.json`  
Sealed split summary: `reports/split-summary.json`

## Scope control

This report profiles only the 38,122 development rows identified by the sealed manifest. The 9,531 final-test rows were not included in feature statistics (`test_rows_profiled = 0`). Before profiling, the command verified the manifest SHA-256 against the sealed split summary.

## Class balance

| Class | Meaning | Development rows | Fraction |
|---|---|---:|---:|
| 0 | goodware | 16,878 | 44.27% |
| 1 | malware | 21,244 | 55.73% |

The imbalance is moderate. Stratified group-aware cross-validation is required. Accuracy must not replace AUC as the primary selection metric. Class weighting can be considered inside model tuning, but should not be assumed beneficial without CV evidence.

## Missing and invalid values

- All 22 numeric candidate features have zero missing, non-finite or parse-error values in the development partition.
- `ImportedDlls` and `ImportedSymbols` have no empty values.
- `Identify` is missing for 11,064 development rows (`29.02%`). Treat missing as an explicit categorical value within the fold-local pipeline.
- The label and SHA1 grouping key were validated before splitting.

## Constant columns

Three numeric columns are constant in the development set:

- `Magic = 267`
- `PE_TYPE = 267`
- `SizeOfOptionalHeader = 224`

They contain no discriminative information in this snapshot. A fold-local variance filter is the preferred implementation so the no-leakage rule remains explicit and reproducible.

## Numeric distributions and outliers

Many PE size/address fields are strongly skewed and have large IQR-outlier counts. Examples include:

| Feature | Median | Maximum | IQR outliers |
|---|---:|---:|---:|
| `BaseOfCode` | 4,096 | 228,319,232 | 7,628 |
| `BaseOfData` | 307,200 | 941,510,656 | 4,145 |
| `Size` | 562,176 | 3,986,104,226 | 5,157 |
| `SizeOfImage` | 491,520 | 1,473,761,280 | 4,653 |
| `SizeOfInitializedData` | 74,752 | 4,294,803,968 | 5,030 |
| `TimeDateStamp` | 1,366,993,954 | 3,964,004,123 | 7,997 |

These values are valid finite numbers, not automatically errors. Tree models can generally consume them directly. Logistic Regression and the PyTorch MLP should receive fold-local robust transformation/scaling; optional `log1p` transforms are candidates only for non-negative size/count columns and must be evaluated inside CV.

`NumberOfRvaAndSizes`, `NumberOfSymbols` and `PointerToSymbolTable` also contain extreme maxima. The pipeline should avoid deleting rows using thresholds learned from the full dataset. Clipping/winsorization, if explored, must be fitted on each training fold.

## Categorical and text-derived attributes

| Feature | Development characteristics | Planned handling |
|---|---|---|
| `Identify` | 228 values including missing; missing is most frequent | Fold-local missing category plus one-hot encoding with unknown-value handling |
| `ImportedDlls` | 9,394 unique strings; median 9 tokens; maximum 82 tokens | Token-based hashing/vectorization with bounded dimensionality |
| `ImportedSymbols` | 15,769 unique strings; median 111 tokens; maximum 2,447 tokens and 103,632 characters | Token-based hashing/vectorization with strict dimensionality and request-size limits |

Hashing vectorization is favored for the two import-list fields because it does not build a full vocabulary and safely handles unseen tokens. Model-specific dimensionality and whether text features improve AUC must still be measured. The raw strings must not be sent to the runtime LLM.

## Leakage review

- `Label` is the target and is excluded from all model inputs.
- `SHA1` is used only for stable identity/grouping and is excluded from prediction.
- `FirstSeenDate` is excluded because collection-date ranges differ materially by label.
- Exact duplicates were removed and SHA1 groups cannot cross partitions.
- Learned transformations and feature selection have not been fitted at this stage.
- The final test feature distribution has not been inspected.

## Preprocessing decisions to implement

1. Numeric pipeline: finite-value validation, median imputation as a defensive step, optional model-specific robust scaling, and fold-local variance filtering.
2. `Identify`: explicit missing token plus one-hot encoding with unknown categories ignored.
3. Import text: bounded `HashingVectorizer` token features; compare inclusion/dimensionality using development CV only.
4. Tree models: no mandatory scaling; linear/MLP models: scaled numeric features.
5. Feature selection/dimensionality choices are tuning parameters. The final choice must be frozen before hold-out evaluation.

## Verification

Seventeen standard-library tests passed, including a test that places an extreme value in a synthetic test row and proves it is absent from development statistics.
