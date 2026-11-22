# Final Hold-out Evaluation

This is the one-time evaluation of the frozen `malware-lightgbm-20261028-v1` pipeline. The configuration hash was `1ffc985b1722a65e7944e5d8af91295663403031082dc28b4a197cdb045e4cac` before test access. No test result was used for model selection or tuning.

| Metric | Result |
|---|---:|
| Test rows | 9,531 |
| Test SHA1 groups | 8,742 |
| ROC AUC | 0.995296 |
| Accuracy | 0.971042 |
| Threshold | 0.50 |
| TN | 4,167 |
| FP | 53 |
| FN | 223 |
| TP | 5,088 |

Confusion-matrix rows are true labels and columns are predicted labels, both in `[0, 1]` order. `FN` means actual malware predicted as goodware; `FP` means actual goodware predicted as malware.

The result must not be used to alter the selected model, preprocessing, hyperparameters or threshold.
