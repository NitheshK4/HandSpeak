import os
import json
import numpy as np
import kagglehub
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure backend folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import run_training
import generate_data

def main():
    # Check if local dataset folder exists in workspace
    local_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Sign Language Dataset")
    if os.path.isdir(local_dataset_dir):
        print(f"Using local dataset found at: {local_dataset_dir}")
        dataset_dir = local_dataset_dir
    else:
        print("Step 1: Downloading kabilan03/sign-language-dataset using kagglehub...")
        # Download latest version
        dataset_dir = kagglehub.dataset_download("kabilan03/sign-language-dataset")
        print(f"Dataset downloaded to: {dataset_dir}")
    
    # Identify folders 0-9
    folders = [str(i) for i in range(10)]
    
    # Initialize MediaPipe HandLandmarker Tasks API
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading hand_landmarker.task model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully.")
        
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Ensure baseline dataset exists
    dataset_path = os.path.join("data", "sign_dataset.json")
    if not os.path.exists(dataset_path):
        print("Baseline dataset not found. Generating synthetic dataset first...")
        generate_data.main()
        
    with open(dataset_path, "r") as f:
        existing_data = json.load(f)
        
    # Remove any existing DIGITS culture entries to make it idempotent
    filtered_data = [item for item in existing_data if item.get("culture") != "DIGITS"]
    print(f"Loaded existing dataset. Removed previous DIGITS entries. Base sample count: {len(filtered_data)}")
    
    new_samples_count = 0
    total_images_processed = 0
    
    print("Step 2: Processing images and extracting landmarks...")
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            # Try alternate path structures if nested
            nested_path1 = os.path.join(dataset_dir, "Dataset", folder)
            nested_path2 = os.path.join(dataset_dir, "Sign Language Dataset", folder)
            if os.path.isdir(nested_path1):
                folder_path = nested_path1
            elif os.path.isdir(nested_path2):
                folder_path = nested_path2
            else:
                print(f"Warning: Directory for digit '{folder}' not found at {folder_path}")
                continue
                
        # List all image files
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing digit '{folder}': found {len(image_files)} images...")
        
        detected_count = 0
        for img_name in image_files:
            total_images_processed += 1
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Load the input image from a file using MediaPipe Image
                mp_image = mp.Image.create_from_file(img_path)
                
                # Detect hand landmarks from the input image
                detection_result = detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    landmarks_list = []
                    for lm in hand_landmarks:
                        landmarks_list.append([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
                        
                    filtered_data.append({
                        "culture": "DIGITS",
                        "label": folder,
                        "landmarks": landmarks_list
                    })
                    detected_count += 1
                    new_samples_count += 1
            except Exception as e:
                # Silently skip errors on individual images
                continue
                
        print(f"Digit '{folder}': successfully extracted landmarks from {detected_count}/{len(image_files)} images.")
        
    detector.close()
    
    print(f"\nExtraction complete! Processed {total_images_processed} images total. Detected hands in {new_samples_count} images.")
    
    # Save the updated dataset
    with open(dataset_path, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Updated dataset saved to {dataset_path} (new total sample count: {len(filtered_data)})")
    
    # Step 3: Trigger training
    print("\nStep 3: Retraining the HandSpeak GNN, MLP, and Random Forest models...")
    run_training()
    print("\nAll models successfully retrained and saved with the new DIGITS dataset!")

if __name__ == "__main__":
    main()
