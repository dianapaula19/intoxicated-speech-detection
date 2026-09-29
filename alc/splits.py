"""Speaker-independent data splits.

ALC has many recordings per speaker, both sober and intoxicated. A random
per-recording split puts the same voices in train and test, so the model can
learn to recognise speakers instead of intoxication. All splits here keep every
speaker on one side only.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def speaker_split(y, speakers, test_size=0.2, seed=42):
    """Split indices into (train, test) with no speaker in both.

    Uses a stratified group k-fold (k = round(1 / test_size)) and takes the
    first fold as the test set, so the class balance of the test set stays
    close to the full data.
    """
    n_splits = max(2, int(round(1 / test_size)))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(folds.split(np.zeros(len(y)), y, groups=speakers))
    return train_idx, test_idx


def speaker_folds(y, speakers, n_splits=5, seed=42):
    """Yield (train, test) index pairs for speaker-independent cross-validation."""
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from folds.split(np.zeros(len(y)), y, groups=speakers)
