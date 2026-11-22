# Tuning Budget Revisions

## 2026-10-28 — Random Forest candidate 2

- Original candidate: 300 trees, unlimited depth, minimum leaf size 5, `max_features=0.5`, balanced subsample weights.
- Observation: the first formal fold did not complete after more than two minutes on the 24 GB M4 development machine, while the first Random Forest candidate completed folds in a practical bounded time.
- Action: interrupted before any candidate-2 fold result was recorded. Reduced to 250 trees and `max_features="sqrt"`; retained unlimited depth, leaf size 5 and balanced subsample weights so the candidate still tests the intended regularization/class-weight contrast.
- Integrity: no final-test data was accessed, no completed metric was discarded, and previously completed candidates remain unchanged. The revised parameters produce a new configuration hash.
