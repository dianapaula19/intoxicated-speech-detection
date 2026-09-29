"""Turning the raw ALC folder into one feature file, and loading it back."""

import json
from pathlib import Path

import numpy as np

from .annotations import annotation_path, intoxication_label, read_metadata
from .features import FeatureConfig, mfcc_from_file

# Only the close-talk headset channel is used (files ending in ``_h_00.wav``).
WAV_PATTERN = "*_h_00.wav"


def find_recordings(root):
    """Sorted list of (wav, annotation) pairs that have both files."""
    pairs = []
    for wav in sorted(Path(root).rglob(WAV_PATTERN)):
        annot = annotation_path(wav)
        if annot.exists():
            pairs.append((wav, annot))
    return pairs


def build(root, output, config=FeatureConfig(), log=print):
    """Extract features for every labelled recording under ``root``.

    Writes ``output`` (a ``.npz`` with arrays ``X``, ``y``, ``speaker``,
    ``utterance``) and a ``<output>.metadata.jsonl`` file with the full
    annotation of each row, in the same order.
    """
    features, labels, speakers, utterances, metadata = [], [], [], [], []
    skipped = 0
    for wav, annot in find_recordings(root):
        meta = read_metadata(annot)
        label = intoxication_label(meta)
        if label is None or "spn" not in meta:
            skipped += 1
            continue
        features.append(mfcc_from_file(wav, config))
        labels.append(label)
        speakers.append(meta["spn"])
        utterances.append(wav.stem)
        metadata.append(meta)

    if not features:
        raise SystemExit(f"No labelled recordings found under {root}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.stack(features),
        y=np.array(labels, dtype=np.int8),
        speaker=np.array(speakers),
        utterance=np.array(utterances),
        config=json.dumps(config.__dict__),
    )
    with open(metadata_path(output), "w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")
    log(f"Saved {len(features)} recordings from {len(set(speakers))} speakers "
        f"to {output} ({skipped} skipped without label)")
    return output


def metadata_path(features_path):
    features_path = Path(features_path)
    return features_path.with_name(features_path.stem + ".metadata.jsonl")


def load(path):
    """Return (X, y, speakers, utterances) from a file written by :func:`build`."""
    data = np.load(path)
    return data["X"], data["y"].astype(int), data["speaker"], data["utterance"]


def load_metadata(path):
    with open(metadata_path(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]
