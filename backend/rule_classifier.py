"""
Rule-Based Hand Gesture Classifier
Uses finger extension/curl states computed from MediaPipe landmarks.
Works for ALL cultures without needing ML training.

Finger Extension Logic:
  - A finger is "extended" if its TIP is farther from the wrist than its PIP joint
  - Thumb uses a horizontal displacement rule instead (since it moves sideways)
"""

import numpy as np


# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def get_finger_states(landmarks):
    """
    Compute finger extension states from a single set of 21 landmarks.
    landmarks: list of [x, y, z] or array of shape (21, 3)

    Returns a dict with keys: thumb, index, middle, ring, pinky
    Each value is True (extended) or False (curled)
    """
    lm = np.array(landmarks)
    wrist = lm[WRIST]

    def dist2d(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    # Index/Middle/Ring/Pinky: tip is extended if farther from wrist than PIP
    index_ext  = dist2d(lm[INDEX_TIP],  wrist) > dist2d(lm[INDEX_PIP],  wrist)
    middle_ext = dist2d(lm[MIDDLE_TIP], wrist) > dist2d(lm[MIDDLE_PIP], wrist)
    ring_ext   = dist2d(lm[RING_TIP],   wrist) > dist2d(lm[RING_PIP],   wrist)
    pinky_ext  = dist2d(lm[PINKY_TIP],  wrist) > dist2d(lm[PINKY_PIP],  wrist)

    # Thumb: use multiple heuristics for robustness
    # Rule 1: tip is above IP joint vertically (pointing up)
    thumb_tip_above_ip = lm[THUMB_TIP][1] < lm[THUMB_IP][1] - 0.015
    # Rule 2: tip is far from index MCP horizontally (pointing sideways)
    thumb_tip_far_from_index = dist2d(lm[THUMB_TIP], lm[INDEX_MCP]) > dist2d(lm[THUMB_IP], lm[INDEX_MCP]) * 1.1
    # Rule 3: tip is far from the wrist (not tucked under)
    thumb_tip_extended = dist2d(lm[THUMB_TIP], wrist) > dist2d(lm[THUMB_IP], wrist)

    thumb_ext = thumb_tip_above_ip or thumb_tip_far_from_index or thumb_tip_extended

    return {
        "thumb":  thumb_ext,
        "index":  index_ext,
        "middle": middle_ext,
        "ring":   ring_ext,
        "pinky":  pinky_ext
    }


def get_advanced_metrics(landmarks):
    """
    Compute additional geometric metrics for disambiguation.
    """
    lm = np.array(landmarks)
    wrist = lm[WRIST]
    
    def dist2d(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # Thumb-Index distance (for OK sign detection)
    thumb_index_dist = dist2d(lm[THUMB_TIP], lm[INDEX_TIP])
    
    # Thumb direction: positive = pointing up, negative = pointing down
    # Compare thumb tip Y to wrist Y (smaller Y = higher on screen)
    thumb_points_up = lm[THUMB_TIP][1] < lm[WRIST][1]
    thumb_points_down = lm[THUMB_TIP][1] > lm[WRIST][1] + 0.1
    
    # Wrist Y reference
    wrist_y = wrist[1]
    
    # Check if thumb tip is above or below knuckle line (MCP joints)
    avg_mcp_y = np.mean([lm[INDEX_MCP][1], lm[MIDDLE_MCP][1], lm[RING_MCP][1], lm[PINKY_MCP][1]])
    thumb_above_knuckles = lm[THUMB_TIP][1] < avg_mcp_y
    
    # Hand scale: wrist to middle MCP distance
    scale = dist2d(lm[MIDDLE_MCP], wrist)
    
    return {
        "thumb_index_dist": thumb_index_dist,
        "thumb_points_up": thumb_points_up,
        "thumb_points_down": thumb_points_down,
        "thumb_above_knuckles": thumb_above_knuckles,
        "scale": scale,
        "wrist_y": wrist_y
    }


def classify_gesture(landmarks, culture="ALL"):
    """
    Rule-based gesture classifier.
    
    Returns dict with:
        label: gesture name
        culture: culture code
        confidence: confidence score (0-1)
        finger_states: dict of finger extension states
    """
    states = get_finger_states(landmarks)
    metrics = get_advanced_metrics(landmarks)
    
    t = states["thumb"]
    i = states["index"]
    m = states["middle"]
    r = states["ring"]
    p = states["pinky"]
    
    thumb_up = metrics["thumb_points_up"]
    thumb_down = metrics["thumb_points_down"]
    thumb_index_dist = metrics["thumb_index_dist"]
    scale = metrics["scale"]
    
    # Normalized thumb-index distance (relative to hand size)
    norm_ti_dist = thumb_index_dist / scale if scale > 0 else thumb_index_dist
    
    def dist2d(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # ─── Match by fingerstate pattern ───────────────────────────────────────
    
    # Spock Salute: All 5 fingers extended, but middle-ring distance is large
    # and index-middle & ring-pinky distances are small.
    if t and i and m and r and p:
        lm = np.array(landmarks)
        index_middle_dist = dist2d(lm[INDEX_TIP], lm[MIDDLE_TIP])
        middle_ring_dist = dist2d(lm[MIDDLE_TIP], lm[RING_TIP])
        ring_pinky_dist = dist2d(lm[RING_TIP], lm[PINKY_TIP])
        
        norm_mr = middle_ring_dist / scale if scale > 0 else 0
        norm_im = index_middle_dist / scale if scale > 0 else 0
        norm_rp = ring_pinky_dist / scale if scale > 0 else 0
        
        if norm_mr > 0.52 and norm_mr > norm_im * 1.4 and norm_mr > norm_rp * 1.1:
            return _result("Spock", "UNIVERSAL", 0.94, states)
            
    # ALL 5 extended = Open Palm / Hello / Namaste
    if t and i and m and r and p:
        if culture in ("ISL",):
            return _result("Namaste", "ISL", 0.92, states)
        elif culture in ("BSL",):
            return _result("Hello", "BSL", 0.90, states)
        elif culture in ("LSE",):
            return _result("Hola", "LSE", 0.92, states)
        return _result("Hello", "ASL", 0.92, states)
    
    # ALL curled = Fist / Yes
    if not t and not i and not m and not r and not p:
        if culture in ("ISL",):
            return _result("Yes", "ISL", 0.90, states)
        elif culture in ("LSE",):
            return _result("Si", "LSE", 0.90, states)
        return _result("Yes", "ASL", 0.90, states)
    
    # Thumb up, others curled = Thumbs Up / Like / Awesome / TheekHai
    if t and not i and not m and not r and not p and thumb_up:
        if culture in ("ISL",):
            return _result("TheekHai", "ISL", 0.95, states)
        elif culture in ("BSL",):
            return _result("Awesome", "BSL", 0.95, states)
        return _result("Like", "UNIVERSAL", 0.95, states)
    
    # Thumb down, others curled = Thumbs Down / Dislike
    if t and not i and not m and not r and not p and thumb_down:
        if culture in ("ISL",):
            return _result("PasandNahi", "ISL", 0.93, states)
        elif culture in ("BSL",):
            return _result("Dislike", "BSL", 0.93, states)
        elif culture in ("LSE",):
            return _result("No", "LSE", 0.93, states)
        return _result("Dislike", "UNIVERSAL", 0.93, states)
    
    # Index + Middle extended = Peace / Victory / Vijay
    if not t and i and m and not r and not p:
        if culture in ("ISL",):
            return _result("Vijay", "ISL", 0.93, states)
        elif culture in ("BSL",):
            return _result("Peace", "BSL", 0.93, states)
        return _result("Peace", "UNIVERSAL", 0.93, states)
    
    # Index only extended = Point / Salute
    if not t and i and not m and not r and not p:
        if culture in ("BSL",):
            return _result("Salute", "BSL", 0.92, states)
        return _result("Point", "ASL", 0.92, states)
    
    # OK sign = thumb + index form circle, others extended
    if norm_ti_dist < 0.35 and m and r and p:
        if culture in ("ISL",):
            return _result("Achaa", "ISL", 0.90, states)
        elif culture in ("BSL",):
            return _result("Perfect", "BSL", 0.90, states)
        return _result("OK", "ASL", 0.90, states)
    
    # ILY (I Love You) = Thumb + Index + Pinky extended
    if t and i and not m and not r and p:
        if culture in ("LSE",):
            return _result("Gracias", "LSE", 0.91, states)
        return _result("Love", "ASL", 0.91, states)
    
    # Call sign = Thumb + Pinky extended (shaka / hang loose)
    if t and not i and not m and not r and p:
        return _result("Call", "UNIVERSAL", 0.90, states)
    
    # Rock sign = Index + Pinky extended (devil horns)
    if not t and i and not m and not r and p:
        return _result("Rock", "UNIVERSAL", 0.91, states)
    
    # Middle finger only
    if not t and not i and m and not r and not p:
        return _result("MidFinger", "UNIVERSAL", 0.92, states)
    
    # Pinky only
    if not t and not i and not m and not r and p:
        return _result("Pinkie", "UNIVERSAL", 0.88, states)
    
    # Gun: Index extended + Thumb up, others curled
    if t and i and not m and not r and not p and thumb_up:
        return _result("Gun", "UNIVERSAL", 0.89, states)
    
    # Four fingers = index + middle + ring + pinky extended, thumb curled
    if not t and i and m and r and p:
        return _result("Four", "UNIVERSAL", 0.90, states)
    
    # Three fingers = index + middle + ring extended
    if not t and i and m and r and not p:
        return _result("Three", "UNIVERSAL", 0.90, states)
    
    # Two fingers = index + middle (same as peace but without culture filter)
    if not t and i and m and not r and not p:
        return _result("TwoUp", "UNIVERSAL", 0.88, states)
    
    # One finger = index only (same as point)
    if not t and i and not m and not r and not p:
        return _result("One", "UNIVERSAL", 0.88, states)
    
    # Palm = all extended (generic fallback for "ALL" culture)
    if i and m and r and p:
        return _result("Palm", "UNIVERSAL", 0.75, states)
    
    # Fallback
    return _result("Unknown", "UNIVERSAL", 0.10, states)


def _result(label, culture, confidence, finger_states):
    fingers_str = []
    if finger_states["thumb"]:  fingers_str.append("thumb")
    if finger_states["index"]:  fingers_str.append("index")
    if finger_states["middle"]: fingers_str.append("middle")
    if finger_states["ring"]:   fingers_str.append("ring")
    if finger_states["pinky"]:  fingers_str.append("pinky")
    
    return {
        "label": label,
        "culture": culture,
        "class": f"{culture}_{label}",
        "confidence": confidence,
        "extended_fingers": fingers_str,
        "latency_ms": 0.0
    }
