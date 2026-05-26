#!/usr/bin/env python3
"""
HandSpeak — Quick Start Script
Automatically generates dataset, trains models, and starts the server.
Run: python3 start.py
"""

import os
import sys
import subprocess

BANNER = """
╔══════════════════════════════════════════════════════╗
║   🤟  HandSpeak — AI Sign Language Recognition       ║
║   Real-time · Multicultural · Rule-Based + GNN       ║
╚══════════════════════════════════════════════════════╝
"""

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(1)

def main():
    print(BANNER)
    base = os.path.dirname(os.path.abspath(__file__))

    # 1. Check dataset
    dataset = os.path.join(base, "data", "sign_dataset.json")
    if not os.path.exists(dataset):
        print("📦 Generating synthetic sign dataset...")
        run(f"python3 backend/generate_data.py", cwd=base)
        print("✅ Dataset ready!\n")
    else:
        print("✅ Dataset already exists — skipping generation.\n")

    # 2. Check models
    gcn = os.path.join(base, "backend", "gcn_model.pth")
    if not os.path.exists(gcn):
        print("🧠 Training GNN, MLP, and Random Forest models...")
        print("   (This takes ~2-3 minutes on first run)\n")
        run(f"python3 backend/train.py", cwd=base)
        print("✅ All models trained!\n")
    else:
        print("✅ Trained models found — skipping training.\n")

    # 3. Start server
    print("🚀 Starting HandSpeak server at http://localhost:8000 ...")
    print("   Press Ctrl+C to stop.\n")
    try:
        run(f"python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload", cwd=base)
    except KeyboardInterrupt:
        print("\n👋 HandSpeak server stopped. See you next time!")

if __name__ == "__main__":
    main()
