import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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

    return np.stack(feature_list), np.array(label_list)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MediaPipe Urdu sign classifier.")
    parser.add_argument("dataset_dir", help="Directory containing landmark JSON files")
    parser.add_argument("--model", default="frontend/model/urdu10_model.h5", help="Trained model path")
    args = parser.parse_args()

    X, y = load_dataset(args.dataset_dir)
    encoder = LabelEncoder()
    encoder.fit(CLASS_LABELS)
    y_encoded = encoder.transform(y)

    model = tf.keras.models.load_model(args.model)
    predictions = model.predict(X, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(y_encoded, predicted_labels, target_names=CLASS_LABELS, digits=4)
    matrix = confusion_matrix(y_encoded, predicted_labels)

    print("=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(matrix)

if __name__ == "__main__":
    main()
