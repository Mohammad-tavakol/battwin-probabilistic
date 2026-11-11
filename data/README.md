# Data partitions

`stages/<process>/train.csv`, `validation.csv` and `test.csv` are the supplied process partitions. Repetitions and partition membership are fixed. `inventory.json` records their paths, counts and checksums; `configs/processes/` defines model inputs, outputs and grouping features.

`condition_id` and `group_id` identify conditions and split groups. `_source_row` is a stable observation identifier, not a position in a partition CSV. `_source_file` identifies the included partition file. Source-study labels in `Paper`, where supplied, are retained for attribution and diagnostic grouping; they are not model predictors.
