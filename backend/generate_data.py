import json
import os
import math
import numpy as np

# Make output directory
os.makedirs("data", exist_ok=True)

# Define templates for 21 joints of the hand (x, y, z)
# Joint indices in MediaPipe:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, Tip)
# 5-8: Index (MCP, PIP, DIP, Tip)
# 9-12: Middle (MCP, PIP, DIP, Tip)
# 13-16: Ring (MCP, PIP, DIP, Tip)
# 17-20: Pinky (MCP, PIP, DIP, Tip)

def get_template_open_palm():
    # Hand open, fingers pointing up
    landmarks = np.zeros((21, 3))
    # Wrist
    landmarks[0] = [0.5, 0.9, 0.0]
    
    # Thumb (curving out to the left)
    landmarks[1] = [0.42, 0.85, -0.01]
    landmarks[2] = [0.35, 0.80, -0.02]
    landmarks[3] = [0.30, 0.76, -0.03]
    landmarks[4] = [0.26, 0.73, -0.04]
    
    # Index
    landmarks[5] = [0.44, 0.65, -0.01]
    landmarks[6] = [0.43, 0.50, -0.02]
    landmarks[7] = [0.42, 0.40, -0.03]
    landmarks[8] = [0.41, 0.30, -0.04]
    
    # Middle
    landmarks[9] = [0.50, 0.62, 0.0]
    landmarks[10] = [0.50, 0.46, -0.01]
    landmarks[11] = [0.50, 0.35, -0.02]
    landmarks[12] = [0.50, 0.24, -0.03]
    
    # Ring
    landmarks[13] = [0.56, 0.65, -0.01]
    landmarks[14] = [0.57, 0.50, -0.02]
    landmarks[15] = [0.58, 0.40, -0.03]
    landmarks[16] = [0.59, 0.31, -0.04]
    
    # Pinky
    landmarks[17] = [0.62, 0.70, -0.02]
    landmarks[18] = [0.64, 0.58, -0.03]
    landmarks[19] = [0.65, 0.50, -0.04]
    landmarks[20] = [0.66, 0.42, -0.05]
    
    return landmarks

def get_template_fist():
    # Closed fist, fingers curled down towards wrist
    landmarks = np.zeros((21, 3))
    landmarks[0] = [0.5, 0.9, 0.0]
    
    # Thumb curled across fingers
    landmarks[1] = [0.43, 0.85, -0.01]
    landmarks[2] = [0.38, 0.82, -0.02]
    landmarks[3] = [0.42, 0.78, -0.03]
    landmarks[4] = [0.48, 0.77, -0.04]
    
    # Index curled
    landmarks[5] = [0.44, 0.65, -0.01]
    landmarks[6] = [0.43, 0.73, -0.03]
    landmarks[7] = [0.45, 0.78, -0.04]
    landmarks[8] = [0.48, 0.76, -0.05]
    
    # Middle curled
    landmarks[9] = [0.50, 0.62, 0.0]
    landmarks[10] = [0.50, 0.72, -0.02]
    landmarks[11] = [0.50, 0.77, -0.03]
    landmarks[12] = [0.49, 0.75, -0.04]
    
    # Ring curled
    landmarks[13] = [0.56, 0.65, -0.01]
    landmarks[14] = [0.56, 0.73, -0.03]
    landmarks[15] = [0.54, 0.77, -0.04]
    landmarks[16] = [0.51, 0.75, -0.05]
    
    # Pinky curled
    landmarks[17] = [0.62, 0.70, -0.02]
    landmarks[18] = [0.60, 0.76, -0.04]
    landmarks[19] = [0.57, 0.79, -0.05]
    landmarks[20] = [0.54, 0.77, -0.06]
    
    return landmarks

def get_template_index_pointing():
    # Index extended, other fingers curled
    landmarks = get_template_fist()
    # Replace index finger with extended index from open palm
    open_p = get_template_open_palm()
    for idx in [5, 6, 7, 8]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_victory():
    # Index and middle extended, ring and pinky curled
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [5, 6, 7, 8, 9, 10, 11, 12]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_thumbs_up():
    # Thumb pointing straight up, fingers folded flat
    landmarks = np.zeros((21, 3))
    landmarks[0] = [0.5, 0.9, 0.0]
    
    # Thumb pointing up
    landmarks[1] = [0.42, 0.82, -0.01]
    landmarks[2] = [0.38, 0.72, -0.02]
    landmarks[3] = [0.37, 0.62, -0.03]
    landmarks[4] = [0.36, 0.52, -0.04]
    
    # Rest of the fingers are curled tightly into the palm
    # Index curled
    landmarks[5] = [0.45, 0.75, -0.02]
    landmarks[6] = [0.52, 0.76, -0.04]
    landmarks[7] = [0.55, 0.77, -0.05]
    landmarks[8] = [0.53, 0.78, -0.05]
    
    # Middle curled
    landmarks[9] = [0.50, 0.75, -0.02]
    landmarks[10] = [0.56, 0.76, -0.04]
    landmarks[11] = [0.58, 0.77, -0.05]
    landmarks[12] = [0.56, 0.78, -0.05]
    
    # Ring curled
    landmarks[13] = [0.55, 0.76, -0.02]
    landmarks[14] = [0.60, 0.77, -0.04]
    landmarks[15] = [0.61, 0.78, -0.05]
    landmarks[16] = [0.58, 0.79, -0.05]
    
    # Pinky curled
    landmarks[17] = [0.60, 0.78, -0.02]
    landmarks[18] = [0.64, 0.79, -0.04]
    landmarks[19] = [0.64, 0.80, -0.05]
    landmarks[20] = [0.61, 0.81, -0.05]
    
    return landmarks

def get_template_thumbs_down():
    # Rotate thumbs up template by 180 degrees around Z axis (pi radians)
    # to make the thumb point straight down
    landmarks = get_template_thumbs_up()
    center = landmarks[9].copy() # Rotate around middle MCP
    shifted = landmarks - center
    cos_theta = math.cos(math.pi)
    sin_theta = math.sin(math.pi)
    # Z rotation matrix applied to X and Y
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    rotated = shifted @ Rz
    return rotated + center

def get_template_ok_sign():
    # Start with open palm template
    landmarks = get_template_open_palm()
    
    # Position thumb tip (4) and index tip (8) to touch
    touch_point = [0.42, 0.60, -0.02]
    landmarks[4] = touch_point
    landmarks[8] = touch_point
    
    # Thumb joints curved in a loop
    landmarks[1] = [0.45, 0.80, -0.01]
    landmarks[2] = [0.41, 0.73, -0.02]
    landmarks[3] = [0.40, 0.66, -0.03]
    
    # Index finger joints curved to meet thumb
    landmarks[5] = [0.44, 0.65, -0.01]
    landmarks[6] = [0.38, 0.58, -0.02]
    landmarks[7] = [0.38, 0.54, -0.03]
    
    # Middle, Ring, Pinky remain extended straight up
    return landmarks

def get_template_ily_sign():
    # I Love You sign: Thumb, Index, and Pinky extended; Middle and Ring curled
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    
    # Extend Thumb
    for idx in [1, 2, 3, 4]:
        landmarks[idx] = open_p[idx]
        
    # Extend Index
    for idx in [5, 6, 7, 8]:
        landmarks[idx] = open_p[idx]
        
    # Extend Pinky
    for idx in [17, 18, 19, 20]:
        landmarks[idx] = open_p[idx]
        
    return landmarks

def get_template_call():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [1, 2, 3, 4, 17, 18, 19, 20]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_four():
    landmarks = get_template_open_palm()
    fist = get_template_fist()
    for idx in [1, 2, 3, 4]:
        landmarks[idx] = fist[idx]
    return landmarks

def get_template_one():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [5, 6, 7, 8]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_rock():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [5, 6, 7, 8, 17, 18, 19, 20]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_pinkie():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [17, 18, 19, 20]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_three():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_two_up():
    landmarks = get_template_fist()
    # Index and Middle close together
    landmarks[5] = [0.47, 0.65, -0.01]
    landmarks[6] = [0.47, 0.50, -0.02]
    landmarks[7] = [0.47, 0.40, -0.03]
    landmarks[8] = [0.47, 0.30, -0.04]
    landmarks[9] = [0.51, 0.62, 0.0]
    landmarks[10] = [0.51, 0.46, -0.01]
    landmarks[11] = [0.51, 0.35, -0.02]
    landmarks[12] = [0.51, 0.24, -0.03]
    return landmarks

def get_template_mid_finger():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    for idx in [9, 10, 11, 12]:
        landmarks[idx] = open_p[idx]
    return landmarks

def get_template_gun():
    landmarks = get_template_fist()
    open_p = get_template_open_palm()
    thumbs_up = get_template_thumbs_up()
    for idx in [5, 6, 7, 8]:
        landmarks[idx] = open_p[idx]
    for idx in [1, 2, 3, 4]:
        landmarks[idx] = thumbs_up[idx]
    return landmarks

def get_template_grabbing():
    landmarks = get_template_open_palm()
    wrist = landmarks[0]
    # Pull joints slightly towards center and bend tips in Z
    for idx in range(1, 21):
        landmarks[idx] = (landmarks[idx] - wrist) * 0.85 + wrist
        # Add bending in Z (depth)
        if idx in [4, 8, 12, 16, 20]:
            landmarks[idx][2] -= 0.08
        elif idx in [3, 7, 11, 15, 19]:
            landmarks[idx][2] -= 0.05
    return landmarks

def get_template_grip():
    landmarks = get_template_fist()
    touch_pt = [0.42, 0.70, -0.05]
    landmarks[4] = touch_pt
    landmarks[8] = touch_pt
    landmarks[12] = touch_pt
    landmarks[16] = touch_pt
    landmarks[20] = touch_pt
    return landmarks

# Define cultures and gesture mapping
# We'll map the templates to actual sign meanings per culture
CULTURE_SIGNS = {
    "ASL": {
        "Hello": get_template_open_palm,
        "Yes": get_template_fist,
        "No": get_template_thumbs_down,
        "Point": get_template_index_pointing,
        "OK": get_template_ok_sign,
        "Love": get_template_ily_sign
    },
    "ISL": {
        "Namaste": get_template_open_palm,
        "Alvida": get_template_open_palm, # Goodbye wave
        "TheekHai": get_template_thumbs_up,
        "PasandNahi": get_template_thumbs_down,
        "Vijay": get_template_victory,
        "Achaa": get_template_ok_sign # Great/Achaa
    },
    "BSL": {
        "Salute": get_template_index_pointing,
        "Awesome": get_template_thumbs_up,
        "Dislike": get_template_thumbs_down,
        "Peace": get_template_victory,
        "Perfect": get_template_ok_sign
    },
    "UNIVERSAL": {
        "Call": get_template_call,
        "Dislike": get_template_thumbs_down,
        "Fist": get_template_fist,
        "Four": get_template_four,
        "Like": get_template_thumbs_up,
        "OK": get_template_ok_sign,
        "One": get_template_one,
        "Palm": get_template_open_palm,
        "Peace": get_template_victory,
        "Rock": get_template_rock,
        "Point": get_template_index_pointing,
        "Pinkie": get_template_pinkie,
        "Three": get_template_three,
        "TwoUp": get_template_two_up,
        "MidFinger": get_template_mid_finger,
        "Gun": get_template_gun,
        "Grabbing": get_template_grabbing,
        "Grip": get_template_grip
    }
}

def rotate_landmarks_3d(landmarks, ax, ay, az):
    """
    Rotate 3D points by angles (in radians) around X, Y, Z axes.
    """
    # X rotation matrix
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)]
    ])
    # Y rotation matrix
    Ry = np.array([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])
    # Z rotation matrix
    Rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation around center of hand (wrist/joint 0 or joint 9)
    center = landmarks[9].copy() # Middle MCP as center of hand
    shifted = landmarks - center
    rotated = shifted @ Rx @ Ry @ Rz
    return rotated + center

def augment_landmarks(template_fn, samples_count=250):
    augmented_data = []
    
    for _ in range(samples_count):
        landmarks = template_fn()
        
        # 1. Scale hand landmarks slightly (wider range for more diversity)
        scale = np.random.uniform(0.80, 1.20)
        center = landmarks[0].copy() # wrist center
        landmarks = (landmarks - center) * scale + center
        
        # 2. Rotate hand in 3D (simulate hand tilt at different angles)
        ax = np.random.uniform(-0.25, 0.25)
        ay = np.random.uniform(-0.25, 0.25)
        az = np.random.uniform(-0.35, 0.35)
        landmarks = rotate_landmarks_3d(landmarks, ax, ay, az)
        
        # 3. Translate hand in frame (simulate different hand positions)
        tx = np.random.uniform(-0.18, 0.18)
        ty = np.random.uniform(-0.18, 0.18)
        tz = np.random.uniform(-0.06, 0.06)
        landmarks += np.array([tx, ty, tz])
        
        # 4. Add Gaussian noise (simulate tracking jitter)
        noise = np.random.normal(0, 0.006, landmarks.shape)
        landmarks += noise
        
        # 5. Horizontal flip (supports both left and right hands)
        if np.random.rand() > 0.5:
            wrist_x = landmarks[0, 0]
            landmarks[:, 0] = 2 * wrist_x - landmarks[:, 0]
        
        # Round to 4 decimal places for clean JSON
        landmarks = np.round(landmarks, 4).tolist()
        augmented_data.append(landmarks)
        
    return augmented_data


def main():
    print("Generating synthetic datasets...")
    dataset = []
    
    # Generate 150 samples for each sign of each culture
    samples_per_class = 150
    
    for culture, signs in CULTURE_SIGNS.items():
        for label, template_fn in signs.items():
            print(f"Generating data for {culture} - {label}...")
            samples = augment_landmarks(template_fn, samples_count=samples_per_class)
            for s in samples:
                dataset.append({
                    "culture": culture,
                    "label": label,
                    "landmarks": s
                })
                
    # Save dataset to JSON file
    output_path = os.path.join("data", "sign_dataset.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset successfully saved to {output_path} ({len(dataset)} total samples).")

if __name__ == "__main__":
    main()
