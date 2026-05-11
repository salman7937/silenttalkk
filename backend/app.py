from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from ultralytics import YOLO  # Added for SilentTalk AI
import os
import cv2
import numpy as np

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load the model you just trained!
# Ensure best.pt is in the same folder as this script
model = YOLO("best.pt") 

# ----------------------------
# SQLite setup
# ----------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
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
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()

# ----------------------------
# SilentTalk AI Endpoint (The Missing Link)
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Convert uploaded image to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Run YOLO11 Inference
    results = model(img, conf=0.5) # 0.5 confidence threshold
    
    if len(results[0].boxes) > 0:
        # Get the label name (e.g., 'aliph')
        class_id = int(results[0].boxes[0].cls)
        label = model.names[class_id]
        return jsonify({'prediction': label})
    
    return jsonify({'prediction': None})

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

if __name__ == '__main__':
    # Changed to 5001 to match your sign.js fetch URL
    app.run(debug=True, port=5001)