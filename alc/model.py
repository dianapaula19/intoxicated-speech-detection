"""Small CNN over MFCC "images" (coefficients x frames)."""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(n_features, n_frames, learning_rate=1e-3):
    """Two conv blocks, global pooling and a sigmoid output.

    Global average pooling instead of Flatten keeps the parameter count small
    (the old model had ~1M weights in its first dense layer for ~12k
    recordings) and makes the model less sensitive to where in the utterance
    a cue occurs.
    """
    model = keras.Sequential([
        keras.Input(shape=(n_features, n_frames, 1)),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
