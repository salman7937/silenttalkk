import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CLASS_LABELS = [
    "aliph",
    "bay",
    "pay",
    "ray",
    "seen",
    "laam",
    "meem",
    "noon",
    "kaaf",
    "hey",
    "none"
]


def distance(a, b):
    return np.linalg.norm(np.array([a["x"], a["y"], a["z"]]) - np.array([b["x"], b["y"], b["z"]]))


def normalize_landmarks(landmarks):
    origin = landmarks[0]
    scale = max(distance(origin, landmarks[9]), 1e-6)
    norm = []
    for l in landmarks:
        norm.extend([ (l["x"] - origin["x"]) / scale,
                      (l["y"] - origin["y"]) / scale,
                      (l["z"] - origin["z"]) / scale ])
    return np.array(norm, dtype=np.float32)


def joint_angle(a, b, c):
    ab = np.array([a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]])
    cb = np.array([c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"]])
    dot = np.dot(ab, cb)
    mag = max(np.linalg.norm(ab) * np.linalg.norm(cb), 1e-6)
    cos_angle = np.clip(dot / mag, -1.0, 1.0)
    return np.arccos(cos_angle)


def extract_features(landmarks):
    if not landmarks or len(landmarks) < 21:
        return None

    normalized = normalize_landmarks(landmarks)
    wrist = landmarks[0]
    tip_indices = [4, 8, 12, 16, 20]

    tip_distances = [distance(wrist, landmarks[idx]) for idx in tip_indices]
    tip_spread = [distance(landmarks[tip_indices[i]], landmarks[tip_indices[i + 1]]) for i in range(len(tip_indices) - 1)]
    finger_angles = [
        joint_angle(landmarks[2], landmarks[3], landmarks[4]),
        joint_angle(landmarks[5], landmarks[6], landmarks[8]),
        joint_angle(landmarks[9], landmarks[10], landmarks[12]),
        joint_angle(landmarks[13], landmarks[14], landmarks[16]),
        joint_angle(landmarks[17], landmarks[18], landmarks[20])
    ]
    palm_distances = [distance(wrist, landmarks[idx]) for idx in tip_indices]

    return np.concatenate([normalized, tip_distances, tip_spread, finger_angles, palm_distances], axis=0)


def load_dataset(dataset_dir):
    feature_list = []
    label_list = []

    for json_path in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            continue
        for item in raw:
            landmarks = item.get("landmarks")
            label = item.get("label")
            if not landmarks or label not in CLASS_LABELS:
                continue
            features = extract_features(landmarks)
            if features is None:
                continue
            feature_list.append(features)
            label_list.append(label)

    if not feature_list:
        raise ValueError("No valid samples found in dataset directory.")

    X = np.stack(feature_list)
    y = np.array(label_list)
    return X, y


def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MediaPipe Urdu sign classifier")
    parser.add_argument("dataset_dir", help="Directory containing landmark JSON files")
    parser.add_argument("--output", default="frontend/model/urdu10_model.h5", help="Path to save the trained model")
    parser.add_argument("--tflite", default="frontend/model/urdu10_model.tflite", help="Path to save the converted TFLite model")
    args = parser.parse_args()

    X, y = load_dataset(args.dataset_dir)
    encoder = LabelEncoder()
    encoder.fit(CLASS_LABELS)
    y_encoded = encoder.transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    model = build_model(X.shape[1], len(CLASS_LABELS))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Saved Keras model to {args.output}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(args.tflite, "wb") as f_out:
        f_out.write(tflite_model)
    print(f"Saved TFLite model to {args.tflite}")

    label_path = os.path.join(os.path.dirname(args.output), "urdu10_labels.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_LABELS, f, ensure_ascii=False, indent=2)
    print(f"Saved label mapping to {label_path}")

if __name__ == "__main__":
    main()
