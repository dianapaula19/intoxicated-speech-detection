"""Parsing of the ALC EMU annotation files (``*_annot.json``)."""

import json
from pathlib import Path

# Annotation fields kept for analysis, with their meaning in the ALC documentation.
FIELDS = {
    "spn": "speaker id",
    "alc": "intoxication class (a = alcoholised, na = sober)",
    "sex": "speaker sex",
    "age": "speaker age",
    "acc": "regional accent",
    "drh": "drinking habits",
    "aak": "breath alcohol concentration (mg/l)",
    "bak": "blood alcohol concentration (per mille)",
    "ges": "speech style / recording condition group",
    "ces": "recording session",
    "wea": "weather during recording",
}


def _coerce(value):
    """Turn numeric strings into floats; keep everything else as-is."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_labels(path):
    """Return the utterance-level labels of an annotation file as a dict."""
    with open(path, encoding="utf-8") as f:
        annotation = json.load(f)
    item = annotation["levels"][0]["items"][0]
    return {label["name"]: label["value"] for label in item["labels"]}


def read_metadata(path):
    """Like :func:`read_labels`, with numeric values converted to floats.

    The speaker id is kept as a zero-padded string so it can be used for
    grouping (``"047"`` and ``"47"`` would otherwise differ).
    """
    labels = read_labels(path)
    metadata = {key: _coerce(value) for key, value in labels.items()}
    if "spn" in labels:
        metadata["spn"] = str(labels["spn"])
    return metadata


def intoxication_label(labels):
    """1 for alcoholised (``alc == "a"``), 0 for sober, None if missing."""
    value = labels.get("alc")
    if value is None:
        return None
    return 1 if value == "a" else 0


def annotation_path(wav_path):
    """Annotation file that belongs to a ``*_h_00.wav`` recording."""
    wav_path = Path(wav_path)
    return wav_path.with_name(wav_path.name.replace(".wav", "_annot.json"))
