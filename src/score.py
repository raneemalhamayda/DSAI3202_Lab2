import json
import os
import numpy as np
import joblib

model = None

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    # model is inside model_output subfolder
    model_path = os.path.join(model_dir, "model_output", "model.pkl")
    if not os.path.exists(model_path):
        # fallback: try direct path
        model_path = os.path.join(model_dir, "model.pkl")
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully.")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["data"])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return {
            "predictions": preds.tolist(),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
