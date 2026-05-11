import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load YOUR custom Urdu model instead of the generic one
# Ensure the 'best.pt' file you downloaded is in this folder
model_path = os.path.join(BASE_DIR, "best.pt") 
model = YOLO(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()
    
    try:
        # Convert incoming image bytes
        img = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 2. Run inference using your 11 custom classes
        results = model(img_cv, conf=0.5) # Increased confidence for better stability
        
        detections = []
        top_prediction = None

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get the Urdu label name (e.g., 'aliph', 'bay')
                label = model.names[class_id]

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                })

        # 3. Get the label with the highest confidence
        if detections:
            # Sort by confidence and pick the best one
            top_prediction = max(detections, key=lambda x: x['confidence'])['label']

        return jsonify({
            "prediction": top_prediction, # This returns 'aliph', 'bay', etc.
            "detections": detections
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Runs on port 5001 for the AI service
    app.run(port=5001, debug=True)