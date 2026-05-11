from ultralytics import YOLO
import os

# 1. Load the YOLO11 Nano base model
model = YOLO("yolo11n.pt") 

# 2. Path to your data.yaml
# Since the file is directly in your backend folder:
data_path = r"C:\Users\aarij\OneDrive\Desktop\JS\Ayesha project\silenttalkk~\silenttalkk\backend\urdu_dataset\data.yaml"

if __name__ == '__main__':
    # 3. Start Training
    model.train(
        data=data_path,
        epochs=50,       # 50 is good for Urdu signs; increase if needed later
        imgsz=640,       # Standard YOLO resolution
        batch=16,        # Safe for your 16GB RAM/i7 setup
        device="cpu",    # Using CPU as we verified earlier
        project="urdu_sign_model", 
        name="urdu_v1"
    )