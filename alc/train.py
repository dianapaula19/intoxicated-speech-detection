"""Train and evaluate the CNN with a speaker-independent split.

    python -m alc.train data/features.npz --out runs/cnn

Writes ``metrics.json``, ``confusion_matrix.png``, ``history.png`` and
``misclassified.csv`` (annotation of every wrong test prediction) to ``--out``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import dataset
from .metrics import summary, uar
from .splits import speaker_split


def standardise(train, *others):
    """Scale every feature row with the training set's mean/std only."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-8
    return [(x - mean) / std for x in (train, *others)]


def class_weights(y):
    """Inverse-frequency weights so both classes count equally in the loss."""
    counts = np.bincount(y, minlength=2)
    return {c: len(y) / (2 * counts[c]) for c in (0, 1) if counts[c]}


def best_threshold(y_true, scores):
    """Decision threshold that maximises UAR on validation data."""
    candidates = np.unique(np.round(scores, 3))
    return float(max(candidates, key=lambda t: uar(y_true, scores >= t)))


def save_plots(out, history, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay(np.array(cm), display_labels=["sober", "intoxicated"]).plot(cmap="Blues")
    plt.title("Test set (unseen speakers)")
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("loss", "auc")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.savefig(out / "history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", help=".npz written by alc.preprocess")
    parser.add_argument("--out", default="runs/cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from .model import build_cnn

    tf.keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, speakers, utterances = dataset.load(args.features)
    train_idx, test_idx = speaker_split(y, speakers, test_size=0.2, seed=args.seed)
    # Validation speakers come out of the training speakers, never the test set.
    fit_rel, val_rel = speaker_split(y[train_idx], speakers[train_idx],
                                     test_size=0.15, seed=args.seed)
    fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]
    assert not set(speakers[fit_idx]) & set(speakers[test_idx])
    assert not set(speakers[val_idx]) & set(speakers[test_idx])

    X_fit, X_val, X_test = standardise(X[fit_idx], X[val_idx], X[test_idx])
    X_fit, X_val, X_test = (x[..., np.newaxis] for x in (X_fit, X_val, X_test))

    model = build_cnn(X.shape[1], X.shape[2])
    history = model.fit(
        X_fit, y[fit_idx],
        validation_data=(X_val, y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(y[fit_idx]),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                 patience=8, restore_best_weights=True)],
        verbose=2,
    )

    threshold = best_threshold(y[val_idx], model.predict(X_val, verbose=0).ravel())
    scores = model.predict(X_test, verbose=0).ravel()
    y_pred = (scores >= threshold).astype(int)
    results = summary(y[test_idx], y_pred)
    results.update(
        threshold=threshold,
        n_train=len(fit_idx), n_val=len(val_idx), n_test=len(test_idx),
        n_speakers_test=len(set(speakers[test_idx])),
    )
    print(json.dumps(results, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    save_plots(out, history.history, results["confusion_matrix"])

    metadata = dataset.load_metadata(args.features)
    wrong = [
        {**metadata[i], "utterance": utterances[i], "true_label": int(y[i]),
         "predicted_label": int(p), "score": float(s)}
        for i, p, s in zip(test_idx, y_pred, scores) if p != y[i]
    ]
    pd.DataFrame(wrong).to_csv(out / "misclassified.csv", index=False)
    model.save(out / "model.keras")


if __name__ == "__main__":
    main()
