"""Offline tests on a tiny synthetic corpus laid out like ALC."""

import json

import numpy as np
import pytest
import soundfile as sf

from alc import annotations, dataset, features, metrics, splits


def _write_recording(folder, utterance, speaker, alc, sr=16_000, seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    pitch = 120 if alc == "na" else 180
    signal = 0.3 * np.sin(2 * np.pi * pitch * t) + 0.01 * rng.standard_normal(t.size)
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{utterance}_h_00.wav", signal.astype(np.float32), sr)
    labels = [{"name": "spn", "value": speaker}, {"name": "alc", "value": alc},
              {"name": "sex", "value": "F"}, {"name": "age", "value": "30"},
              {"name": "bak", "value": "0.5" if alc == "a" else "0.0"}]
    annot = {"levels": [{"items": [{"labels": labels}]}]}
    (folder / f"{utterance}_h_00_annot.json").write_text(json.dumps(annot))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "ALC"
    for s in range(10):
        speaker = f"{s:03d}"
        for k, alc in enumerate(["na", "na", "a"]):
            _write_recording(root / speaker, f"{speaker}{k:07d}", speaker, alc, seed=s * 10 + k)
    return root


def test_annotations_round_trip(corpus):
    annot = sorted(corpus.rglob("*_annot.json"))[0]
    meta = annotations.read_metadata(annot)
    assert meta["spn"] == "000"  # kept as a zero-padded string
    assert meta["age"] == 30.0
    assert annotations.intoxication_label({"alc": "a"}) == 1
    assert annotations.intoxication_label({"alc": "na"}) == 0
    assert annotations.intoxication_label({}) is None


def test_features_shape_and_normalisation():
    config = features.FeatureConfig(n_frames=50)
    signal = np.random.default_rng(0).standard_normal(8_000).astype(np.float32)
    mfcc = features.mfcc_from_signal(signal, config)
    assert mfcc.shape == (config.n_features, 50)
    full = features.cmvn(np.random.default_rng(1).standard_normal((13, 100)) * 5 + 3)
    np.testing.assert_allclose(full.mean(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(full.std(axis=1), 1, atol=1e-6)
    assert features.pad_or_trim(np.ones((2, 3)), 5).shape == (2, 5)
    assert features.pad_or_trim(np.ones((2, 8)), 5).shape == (2, 5)


def test_build_and_speaker_split(corpus, tmp_path):
    out = dataset.build(corpus, tmp_path / "features.npz",
                        features.FeatureConfig(n_frames=40), log=lambda *_: None)
    X, y, speakers, utterances = dataset.load(out)
    assert X.shape == (30, 39, 40)
    assert y.sum() == 10
    assert len(dataset.load_metadata(out)) == 30

    train, test = splits.speaker_split(y, speakers, test_size=0.2)
    assert not set(speakers[train]) & set(speakers[test])
    assert len(train) + len(test) == len(y)
    for train, test in splits.speaker_folds(y, speakers, n_splits=5):
        assert not set(speakers[train]) & set(speakers[test])


def test_uar_ignores_class_balance():
    y_true = [0] * 90 + [1] * 10
    always_sober = [0] * 100
    result = metrics.summary(y_true, always_sober)
    assert result["accuracy"] == 0.9
    assert result["uar"] == 0.5


def test_training_end_to_end(corpus, tmp_path):
    pytest.importorskip("tensorflow")
    from alc import train

    out = dataset.build(corpus, tmp_path / "features.npz",
                        features.FeatureConfig(n_frames=40), log=lambda *_: None)
    run = tmp_path / "run"
    train.main([str(out), "--out", str(run), "--epochs", "2", "--batch-size", "4"])
    result = json.loads((run / "metrics.json").read_text())
    assert 0 <= result["uar"] <= 1
    assert (run / "confusion_matrix.png").exists()
