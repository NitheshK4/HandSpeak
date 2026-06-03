import json
import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Ensure the backend directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import local modules
from models import HandGCN, HandMLP
from train import preprocess_landmarks, run_training
from rule_classifier import classify_gesture
import generate_data

app = FastAPI(
    title="HandSpeak API",
    description="HandSpeak — Real-time multicultural sign language recognition. Supports ASL, ISL, BSL and Universal gestures.",
    version="1.0.0"
)

# Enable CORS to allow direct frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold loaded models
gcn_model = None
mlp_model = None
rf_model = None
class_mapping = None
is_training_active = False

class LandmarkData(BaseModel):
    landmarks: List[List[float]] # List of 21 points, each containing [x, y, z]

class CustomGestureData(BaseModel):
    culture: str
    label: str
    landmarks: List[List[float]]

def load_models():
    global gcn_model, mlp_model, rf_model, class_mapping
    
    mapping_path = "backend/class_mapping.json"
    if not os.path.exists(mapping_path):
        print("Models and mappings not found. Please train models first.")
        return False
        
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
        
    num_classes = len(class_mapping["class_to_idx"])
    
    # Load GCN
    gcn_path = "backend/gcn_model.pth"
    if os.path.exists(gcn_path):
        gcn_model = HandGCN(in_features=3, num_classes=num_classes)
        gcn_model.load_state_dict(torch.load(gcn_path, map_location=torch.device("cpu")))
        gcn_model.eval()
        
    # Load MLP
    mlp_path = "backend/mlp_model.pth"
    if os.path.exists(mlp_path):
        mlp_model = HandMLP(in_features=21*3, num_classes=num_classes)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=torch.device("cpu")))
        mlp_model.eval()
        
    # Load Random Forest
    rf_path = "backend/rf_model.pkl"
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
            
    print("All models successfully loaded!")
    return True

@app.on_event("startup")
def startup_event():
    # 1. Ensure dataset exists
    dataset_path = "data/sign_dataset.json"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating synthetic sign dataset...")
        generate_data.main()
        
    # 2. Check if models are trained, if not train them
    models_ready = load_models()
    if not models_ready:
        print("Training models on startup...")
        run_training()
        load_models()

@app.get("/")
def read_root():
    # Serve index.html from frontend folder
    return FileResponse("frontend/index.html")

def get_culture_predictions(probs, culture, idx_to_class):
    if not culture or culture == "ALL":
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])
        
    culture_probs = {}
    total_culture_prob = 0.0
    
    for idx_str, class_name in idx_to_class.items():
        cls_culture = class_name.split("_")[0]
        if cls_culture.upper() == culture.upper():
            idx = int(idx_str)
            prob = float(probs[idx])
            culture_probs[idx] = prob
            total_culture_prob += prob
            
    if len(culture_probs) == 0:
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])
        
    # Re-normalize conditional probability
    best_idx = None
    best_conf = -1.0
    for idx, prob in culture_probs.items():
        norm_prob = prob / total_culture_prob if total_culture_prob > 0 else 0.0
        if norm_prob > best_conf:
            best_conf = norm_prob
            best_idx = idx
            
    return best_idx, best_conf

@app.post("/api/predict")
def predict_gesture(data: LandmarkData, culture: str = "ALL"):
    global gcn_model, mlp_model, rf_model, class_mapping
    
    landmarks = data.landmarks
    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Must provide exactly 21 landmarks.")
    
    response = {}
    
    # ── 1. Rule-Based Classifier (always runs first, no models needed) ──────
    rule_start = time.perf_counter()
    rule_result = classify_gesture(landmarks, culture)
    rule_time = (time.perf_counter() - rule_start) * 1000  # ms
    rule_result["latency_ms"] = rule_time
    response["rule"] = rule_result

    # ── 2. ML Models (GNN, MLP, RF) ─────────────────────────────────────────
    if gcn_model is None or mlp_model is None or rf_model is None:
        # If models not loaded yet, duplicate rule result for all three slots
        placeholder = dict(rule_result)
        placeholder["latency_ms"] = 0.0
        response["gnn"] = placeholder
        response["mlp"] = placeholder
        response["rf"] = placeholder
        
        print(f"[PREDICTIONS] Rule-Based -> {rule_result['class']} ({rule_result['confidence']*100:.1f}%)", flush=True)
        return response
        
    # Process 2D features: (21, 3) -> (1, 21, 3)
    feat = preprocess_landmarks([landmarks])
    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    
    # Class labels
    idx_to_class = class_mapping["idx_to_class"]
    
    # GNN Prediction
    start_time = time.perf_counter()
    with torch.no_grad():
        gcn_logits = gcn_model(feat_tensor)
        gcn_probs = F.softmax(gcn_logits, dim=1).squeeze().numpy()
    gcn_time = (time.perf_counter() - start_time) * 1000
    
    gcn_idx, gcn_conf = get_culture_predictions(gcn_probs, culture, idx_to_class)
    gcn_class = idx_to_class[str(gcn_idx)]
    gcn_parts = gcn_class.split("_", 1)
    gcn_culture_name, gcn_label = gcn_parts[0], gcn_parts[1] if len(gcn_parts) > 1 else gcn_class
    response["gnn"] = {
        "class": gcn_class,
        "culture": gcn_culture_name,
        "label": gcn_label,
        "confidence": gcn_conf,
        "latency_ms": gcn_time
    }
    
    # MLP Prediction
    start_time = time.perf_counter()
    with torch.no_grad():
        mlp_logits = mlp_model(feat_tensor)
        mlp_probs = F.softmax(mlp_logits, dim=1).squeeze().numpy()
    mlp_time = (time.perf_counter() - start_time) * 1000
    
    mlp_idx, mlp_conf = get_culture_predictions(mlp_probs, culture, idx_to_class)
    mlp_class = idx_to_class[str(mlp_idx)]
    mlp_parts = mlp_class.split("_", 1)
    mlp_culture_name, mlp_label = mlp_parts[0], mlp_parts[1] if len(mlp_parts) > 1 else mlp_class
    response["mlp"] = {
        "class": mlp_class,
        "culture": mlp_culture_name,
        "label": mlp_label,
        "confidence": mlp_conf,
        "latency_ms": mlp_time
    }
    
    # Random Forest Prediction
    start_time = time.perf_counter()
    feat_flat = feat.reshape(1, -1)
    rf_probs = rf_model.predict_proba(feat_flat)[0]
    rf_time = (time.perf_counter() - start_time) * 1000
    
    rf_idx, rf_conf = get_culture_predictions(rf_probs, culture, idx_to_class)
    rf_class = idx_to_class[str(rf_idx)]
    rf_parts = rf_class.split("_", 1)
    rf_culture_name, rf_label = rf_parts[0], rf_parts[1] if len(rf_parts) > 1 else rf_class
    response["rf"] = {
        "class": rf_class,
        "culture": rf_culture_name,
        "label": rf_label,
        "confidence": rf_conf,
        "latency_ms": rf_time
    }
    
    print(f"[PREDICTIONS] Culture: {culture}", flush=True)
    print(f"  Rule -> {rule_result['class']} ({rule_result['confidence']*100:.1f}%)", flush=True)
    print(f"  GNN  -> {gcn_class} ({response['gnn']['confidence']*100:.1f}%)", flush=True)
    print(f"  MLP  -> {mlp_class} ({response['mlp']['confidence']*100:.1f}%)", flush=True)
    print(f"  RF   -> {rf_class} ({response['rf']['confidence']*100:.1f}%)", flush=True)
    
    return response

@app.get("/api/metrics")
def get_metrics():
    metrics_path = "data/training_metrics.json"
    if not os.path.exists(metrics_path):
        return {"trained": False, "message": "No training metrics available. Run training first."}
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["trained"] = True
    metrics["is_training_active"] = is_training_active
    return metrics

def bg_train_task():
    global is_training_active
    try:
        is_training_active = True
        run_training()
        load_models()
    finally:
        is_training_active = False

@app.post("/api/train")
def trigger_training(background_tasks: BackgroundTasks):
    global is_training_active
    if is_training_active:
        return {"status": "already_training", "message": "Training is already in progress."}
        
    background_tasks.add_task(bg_train_task)
    return {"status": "started", "message": "Model training triggered in background."}

@app.post("/api/save_gesture")
def save_gesture(data: CustomGestureData):
    dataset_path = "data/sign_dataset.json"
    
    if not os.path.exists(dataset_path):
        existing_data = []
    else:
        with open(dataset_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
                
    # Add new sample
    new_sample = {
        "culture": data.culture.upper(),
        "label": data.label,
        "landmarks": data.landmarks
    }
    existing_data.append(new_sample)
    
    with open(dataset_path, "w") as f:
        json.dump(existing_data, f, indent=2)
        
    return {"status": "saved", "message": f"Gesture '{data.culture}_{data.label}' recorded. Total samples: {len(existing_data)}."}

@app.get("/api/export_dataset")
def export_dataset():
    dataset_path = "data/sign_dataset.json"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    return FileResponse(
        dataset_path,
        media_type="application/json",
        filename="sign_dataset.json"
    )

@app.get("/api/dataset_stats")
def get_dataset_stats():
    dataset_path = "data/sign_dataset.json"
    if not os.path.exists(dataset_path):
        return {"total_samples": 0, "cultures": {}}
    with open(dataset_path, "r") as f:
        try:
            dataset = json.load(f)
        except Exception:
            return {"total_samples": 0, "cultures": {}}
            
    stats = {}
    for sample in dataset:
        culture = sample.get("culture", "UNIVERSAL").upper()
        label = sample.get("label", "Unknown")
        if culture not in stats:
            stats[culture] = {}
        if label not in stats[culture]:
            stats[culture][label] = 0
        stats[culture][label] += 1
        
    return {
        "total_samples": len(dataset),
        "cultures": stats
    }

@app.get("/api/training_history")
def get_training_history():
    history_path = "data/training_history.json"
    if not os.path.exists(history_path):
        return []
    with open(history_path, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

# Mount frontend files (CSS, JS) so they can be loaded directly from server
# We mount frontend at root /static
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
