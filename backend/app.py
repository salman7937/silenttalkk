from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import json

import numpy as np
import tensorflow as tf

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.abspath(os.path.join(basedir, "..", "frontend"))
model_dir = os.path.join(frontend_dir, "model")
model_path = os.path.join(model_dir, "urdu10_model.h5")
labels_path = os.path.join(model_dir, "urdu10_labels.json")

mediapipe_model = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        mediapipe_labels = json.load(f)
else:
    mediapipe_labels = []

# ----------------------------
# SQLite setup
# ----------------------------
db_path = os.path.join(basedir, 'instance', 'users.db')
if not os.path.exists(os.path.dirname(db_path)):
    os.makedirs(os.path.dirname(db_path))

app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ----------------------------
# Database model (Same as yours)
# ----------------------------
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role     = db.Column(db.String(50),  nullable=False)

    def __init__(self, fullname: str, email: str, password: str, role: str):
        self.fullname = fullname
        self.email    = email
        self.password = password
        self.role     = role

with app.app_context():
    db.create_all()


def _distance(a, b):
    return np.linalg.norm(
        np.array([a["x"], a["y"], a.get("z", 0.0)], dtype=np.float32) -
        np.array([b["x"], b["y"], b.get("z", 0.0)], dtype=np.float32)
    )


def _normalize_landmarks(landmarks):
    origin = landmarks[0]
    scale = max(_distance(origin, landmarks[9]), 1e-6)
    norm = []
    for pt in landmarks:
        norm.extend([
            (pt["x"] - origin["x"]) / scale,
            (pt["y"] - origin["y"]) / scale,
            (pt.get("z", 0.0) - origin.get("z", 0.0)) / scale
        ])
    return np.array(norm, dtype=np.float32)


def _joint_angle(a, b, c):
    ab = np.array([a["x"] - b["x"], a["y"] - b["y"], a.get("z", 0.0) - b.get("z", 0.0)], dtype=np.float32)
    cb = np.array([c["x"] - b["x"], c["y"] - b["y"], c.get("z", 0.0) - b.get("z", 0.0)], dtype=np.float32)
    dot = np.dot(ab, cb)
    mag = max(np.linalg.norm(ab) * np.linalg.norm(cb), 1e-6)
    return float(np.arccos(np.clip(dot / mag, -1.0, 1.0)))


def _extract_features(landmarks):
    if not landmarks or len(landmarks) < 21:
        return None

    normalized = _normalize_landmarks(landmarks)
    wrist = landmarks[0]
    tip_indices = [4, 8, 12, 16, 20]

    tip_distances = [_distance(wrist, landmarks[idx]) for idx in tip_indices]
    tip_spread = [_distance(landmarks[tip_indices[i]], landmarks[tip_indices[i + 1]]) for i in range(len(tip_indices) - 1)]
    finger_angles = [
        _joint_angle(landmarks[2], landmarks[3], landmarks[4]),
        _joint_angle(landmarks[5], landmarks[6], landmarks[8]),
        _joint_angle(landmarks[9], landmarks[10], landmarks[12]),
        _joint_angle(landmarks[13], landmarks[14], landmarks[16]),
        _joint_angle(landmarks[17], landmarks[18], landmarks[20]),
    ]
    palm_distances = [_distance(wrist, landmarks[idx]) for idx in tip_indices]

    return np.concatenate([normalized, tip_distances, tip_spread, finger_angles, palm_distances], axis=0).astype(np.float32)

# ----------------------------
# Routes to serve frontend
# ----------------------------
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ----------------------------
# Auth Endpoints (Same as yours)
# ----------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(email=data.get('email')).first():
        return jsonify({'message': 'User already exists'}), 400
    new_user = User(fullname=data.get('fullname'), email=data.get('email'), 
                    password=data.get('password'), role=data.get('role'))
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data.get('email'), password=data.get('password')).first()
    if user:
        return jsonify({'message': 'Login successful', 'role': user.role}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/predict', methods=['POST'])
def predict_removed():
    return jsonify({
        'error': 'YOLO /predict endpoint has been disabled. Use MediaPipe frontend inference.'
    }), 410


@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    if mediapipe_model is None or not mediapipe_labels:
        return jsonify({'error': 'MediaPipe classifier model or labels are missing.'}), 500

    payload = request.get_json(silent=True) or {}
    landmarks = payload.get('landmarks')
    features = _extract_features(landmarks)
    if features is None:
        return jsonify({'prediction': 'none', 'confidence': 0.0, 'top3': []})

    probs = mediapipe_model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {'label': mediapipe_labels[int(idx)], 'confidence': float(probs[int(idx)])}
        for idx in top_indices
    ]
    best = top3[0]
    return jsonify({
        'prediction': best['label'],
        'confidence': best['confidence'],
        'top3': top3
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
