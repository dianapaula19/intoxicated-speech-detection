# First experiments (Nov–Dec 2024)

The original CNN notebooks, kept for reference. They were run from the repository root on
`Processed_ALC/` and `Processed_Stats_ALC/` (per-recording `.npy` files made by the scripts that
`alc.preprocess` replaced).

| Notebook | Accuracy | Recall (sober) | Recall (intoxicated) | UAR |
|---|---|---|---|---|
| `cnn.ipynb` | 0.68 | 0.87 | 0.28 | 0.58 |
| `cnn-stats.ipynb` | 0.69 | 0.91 | 0.22 | 0.57 |

These numbers are **not** speaker-independent: `train_test_split` split recordings at random, so
the same speakers appear in train and test. The test set was also used as the validation set
during training, and each split was scaled by its own maximum. The `alc` package fixes all three;
use `python -m alc.train` for comparable results.
