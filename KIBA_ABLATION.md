# KIBA ablation records

The four archived training scripts and matching per-epoch curves share the same filename prefix. The summary reports the highest validation concordance index (CI) over 20 epochs for one recorded run per configuration. Each run used a random 85/15 pair split (split seed 42). The 50k condition uses 50,000 pairs. These results describe validation performance only; generalization to unseen proteins requires a separate evaluation.

The KAN variant changes the position-wise MLP in each convolutional branch of the compound and target encoders. The repository includes ongoing work. These archived scripts preserve the original path assumptions and require the project modules and separately obtained input data.
