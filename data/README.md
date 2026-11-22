# Data provenance and handling

## Intended source

- Repository: https://github.com/fabriciojoc/brazilian-malware-dataset
- Repository description: Brazilian malware and goodware dataset associated with Ceschin et al., “The Need for Speed: An Analysis of Brazilian Malware Classifiers” (2018).
- Expected assignment schema: approximately 50,000 records, 27 model input attributes and a `Label` column with `0 = goodware`, `1 = malware`.

## Verified project dataset

The repository archive was manually downloaded on 2026-10-28 because the execution environment could not reach GitHub through the configured local proxy. The project uses only the nested structured file `brazilian-malware.csv`; it does not extract or execute binaries.

| Property | Verified result |
|---|---|
| Local raw file | `data/raw/brazilian-malware.csv` |
| Raw CSV SHA-256 | `d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242` |
| Rows | 50,181 |
| Columns | 28 total: 27 non-label columns plus `Label` |
| Label mapping | `0 = goodware`, `1 = malware` |
| Label counts | 21,116 goodware; 29,065 malware |
| Exact duplicate extra rows | 2,480 |
| Unique SHA1 values | 43,411 |
| SHA1 groups with multiple rows | 3,542 |
| SHA1 groups containing conflicting labels | 18 |

The exact schema and column statistics are recorded in `reports/data-validation.json`. The source archive and nested file hashes are recorded in `data/metadata/source.json`.

## Column roles before model development

- `Label`: target only; never passed to inference.
- `SHA1`: stable record/group identifier; excluded from predictive features. It will be used to prevent the same file identity crossing evaluation partitions.
- `FirstSeenDate`: collection-time metadata and a potential leakage source; excluded from the initial predictive feature set. Any later use would require a documented, development-only experiment and a new leakage review.
- The remaining 25 columns are candidate model inputs. Three are constant in the raw data (`Magic`, `PE_TYPE`, `SizeOfOptionalHeader`) and will be handled inside fold-local preprocessing/feature selection rather than removed using full-dataset learned statistics.
- `Identify`, `ImportedDlls` and `ImportedSymbols` are categorical/text-derived PE attributes. Their encoding strategy must be fit within each training fold.

This means the source dataset matches the assignment's stated 27 non-label attributes, while the defensible initial model schema intentionally excludes two helper/leakage-risk attributes. The difference is explicit rather than hidden.

## Directory design

```text
data/
  README.md                 # provenance, policy and verified facts
  metadata/
    source.json             # source archive/file hashes and retrieval facts
    feature-roles.json      # target, grouping and candidate-input decisions
  raw/
    brazilian-malware.csv   # immutable source CSV; ignored by Git
  splits/
    README.md               # split policy
    ...                     # generated stable IDs/indices; ignored by Git
  sources/
    ...                     # optional third-party checkout; ignored by Git
```

There is deliberately no globally preprocessed training dataset. Learned imputation, encoding, scaling and feature selection must be fitted inside cross-validation folds. Final fitted transformations belong in the versioned production artifact.

## Safety and storage policy

- Use only structured CSV/tabular features required for the assignment.
- Do not obtain or execute malware binaries.
- `data/raw/`, `data/splits/` and third-party source checkouts are ignored by Git by default.
- Record source URL, repository commit/tag, retrieval date, filename, byte size and SHA-256 for every raw input used.
- If the actual repository contents differ from the assignment description, report the differences; do not fabricate or combine data to make it appear compliant.
