# Data split artifacts

This directory stores generated stable row/group identifiers and split manifests. Generated split files are ignored by Git unless a later review explicitly approves a small, non-sensitive manifest.

Split policy:

1. `SHA1` is the group identifier so the same file identity cannot cross development and final-test partitions.
2. The 20% final test partition is created before learned preprocessing or feature selection.
3. Conflicting-label SHA1 groups must be resolved by a documented rule before splitting; they may not be silently assigned.
4. The remaining development partition uses stratified 10-fold evaluation with fold-local preprocessing.
5. The final test set remains sealed until the production configuration is frozen.
