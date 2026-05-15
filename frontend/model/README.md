This folder is reserved for the runtime Urdu sign classifier artifact.

Place the browser-compatible model here as `urdu10_model.tflite` or an equivalent runtime file.

Use the dataset capture page at `frontend/capture.html` to record landmark examples, then train a model with:

```bash
python scripts/train_mediapipe_classifier.py dataset
```

After training, the model will be saved to `frontend/model/urdu10_model.h5` and `frontend/model/urdu10_model.tflite`.

The frontend MediaPipe predictor is implemented in `frontend/mediapipe_predictor.js`, and it will use a loaded model artifact once available.