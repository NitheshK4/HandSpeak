import json
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import model architectures
from models import HandGCN, HandMLP

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_landmarks(landmarks_list):
    """
    Transforms raw landmarks (21, 3) into normalized 2D features (21, 3):
    - Relative 2D coordinates from wrist (x_rel, y_rel) normalized by hand scale
    - Normalized 2D distance to wrist
    """
    processed = []
    for lm in landmarks_list:
        lm_np = np.array(lm) # (21, 3)
        lm_2d = lm_np[:, :2] # Discard Z coordinate for robust inference
        wrist = lm_2d[0]
        
        # Calculate relative coordinates
        rel_lm = lm_2d - wrist # (21, 2)
        
        # Scale normalization: Divide relative coordinates by the distance
        # from wrist (0) to middle finger MCP (9) to make it invariant to hand-to-camera distance
        scale = np.linalg.norm(rel_lm[9])
        if scale > 0:
            rel_lm = rel_lm / scale
        
        # Calculate distance to wrist (now normalized in 2D space)
        dist = np.linalg.norm(rel_lm, axis=1, keepdims=True) # (21, 1)
        
        # Combine to (21, 3)
        feat = np.hstack([rel_lm, dist])
        processed.append(feat)
        
    return np.array(processed)

def load_data(data_path="data/sign_dataset.json"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")
        
    with open(data_path, "r") as f:
        raw_data = json.load(f)
        
    landmarks = []
    labels = []
    
    # Class mapping
    class_to_idx = {}
    idx_to_class = {}
    
    for item in raw_data:
        class_name = f"{item['culture']}_{item['label']}"
        if class_name not in class_to_idx:
            idx = len(class_to_idx)
            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name
            
        landmarks.append(item['landmarks'])
        labels.append(class_to_idx[class_name])
        
    # Preprocess
    X = preprocess_landmarks(landmarks)
    y = np.array(labels)
    
    return X, y, class_to_idx, idx_to_class

def train_pytorch_model(model, train_loader, val_loader, num_classes, epochs=60, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Reduce LR by 50% if val_loss doesn't improve for 8 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()
                
        val_loss = val_running_loss / total_val
        val_acc = correct_val / total_val
        
        # Step LR scheduler on val loss
        scheduler.step(val_loss)
        
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
                  
    return history

def run_training(epochs=100, lr=0.001):
    print("Loading data...")
    X, y, class_to_idx, idx_to_class = load_data()
    
    # Save mapping for server
    os.makedirs("backend", exist_ok=True)
    with open("backend/class_mapping.json", "w") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()}
        }, f, indent=2)
        
    num_classes = len(class_to_idx)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # PyTorch loaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 1. Train Graph Neural Network
    print("\n--- Training Graph Neural Network (GNN) ---")
    gcn_model = HandGCN(in_features=3, num_classes=num_classes)
    start_time = time.time()
    #edited on 04/06/26 by Nithesh kumar
    gcn_history = train_pytorch_model(gcn_model, train_loader, val_loader, num_classes, epochs=epochs, lr=lr)
    gcn_train_time = time.time() - start_time
    
    # Measure GCN inference latency
    gcn_model.eval()
    sample_tensor = torch.tensor(X_val[:100], dtype=torch.float32)
    start_inf = time.time()
    with torch.no_grad():
        _ = gcn_model(sample_tensor)
    gcn_inf_latency = (time.time() - start_inf) / 100 * 1000 # ms per sample
    
    # 2. Train MLP
    print("\n--- Training Multi-Layer Perceptron (MLP) ---")
    mlp_model = HandMLP(in_features=21*3, num_classes=num_classes)
    start_time = time.time()
    #edited on 04/06/26 by Nithesh kumar
    mlp_history = train_pytorch_model(mlp_model, train_loader, val_loader, num_classes, epochs=epochs, lr=lr)
    mlp_train_time = time.time() - start_time
    
    # Measure MLP inference latency
    mlp_model.eval()
    start_inf = time.time()
    with torch.no_grad():
        _ = mlp_model(sample_tensor)
    mlp_inf_latency = (time.time() - start_inf) / 100 * 1000 # ms per sample
    
    # 3. Train Random Forest (Scikit-Learn)
    print("\n--- Training Random Forest (General ML) ---")
    # Flatten inputs for Random Forest
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    start_time = time.time()
    rf_model.fit(X_train_flat, y_train)
    rf_train_time = time.time() - start_time
    
    y_pred_val = rf_model.predict(X_val_flat)
    rf_val_acc = accuracy_score(y_val, y_pred_val)
    print(f"Random Forest Validation Accuracy: {rf_val_acc*100:.2f}%")
    
    # Measure RF inference latency
    start_inf = time.time()
    for i in range(100):
        _ = rf_model.predict(X_val_flat[i:i+1])
    rf_inf_latency = (time.time() - start_inf) / 100 * 1000 # ms per sample
    
    # Save Model Weights
    torch.save(gcn_model.state_dict(), "backend/gcn_model.pth")
    torch.save(mlp_model.state_dict(), "backend/mlp_model.pth")
    with open("backend/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
        
    print("\nModels saved successfully!")
    
    # Compile performance comparison metrics
    metrics = {
        "classes": list(class_to_idx.keys()),
        "gnn": {
            "val_acc": gcn_history["val_acc"][-1],
            "train_time_sec": gcn_train_time,
            "inference_latency_ms": gcn_inf_latency,
            "history": gcn_history
        },
        "mlp": {
            "val_acc": mlp_history["val_acc"][-1],
            "train_time_sec": mlp_train_time,
            "inference_latency_ms": mlp_inf_latency,
            "history": mlp_history
        },
        "rf": {
            "val_acc": float(rf_val_acc),
            "train_time_sec": rf_train_time,
            "inference_latency_ms": rf_inf_latency
        }
    }
    
    with open("data/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Metrics successfully logged to data/training_metrics.json")

    # Append to training history
    history_path = "data/training_history.json"
    history_data = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_data = json.load(f)
        except Exception:
            pass
            
    history_data.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gnn_val_acc": float(gcn_history["val_acc"][-1]),
        "mlp_val_acc": float(mlp_history["val_acc"][-1]),
        "rf_val_acc": float(rf_val_acc)
    })
    
    try:
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        print("Training run successfully logged to history.")
    except Exception as e:
        print(f"Error logging to history: {e}")

if __name__ == "__main__":
    run_training()
