# Improved Face Monitoring Backend with Per-Account Registration
# Better face matching and multiple face detection

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime
import hashlib

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

# Global CORS handler
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Per-account face storage: {voter_id: {features, image, registered_at}}
REGISTERED_FACES = {}

# Current session voter
CURRENT_VOTER_ID = None

# Initialize face detector with multiple cascades for better detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("✓ Face detectors loaded successfully")
except Exception as e:
    print(f"Error loading face detectors: {e}")

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def detect_faces_robust(image):
    """
    Robust face detection with multiple cascade classifiers
    Returns: (status, faces_list)
    - status: 'NO_FACE', 'ONE_FACE', 'MULTIPLE_FACES', 'ERROR'
    - faces_list: list of (x, y, w, h) tuples
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Try multiple detection methods
        all_faces = []
        
        # Method 1: Default cascade
        faces1 = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Method 2: Alt cascade (more sensitive)
        faces2 = face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(80, 80)
        )
        
        # Combine detections
        for (x, y, w, h) in faces1:
            all_faces.append((x, y, w, h))
        
        for (x, y, w, h) in faces2:
            # Check if this face overlaps with existing detections
            is_duplicate = False
            for (ex, ey, ew, eh) in all_faces:
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                face_area = w * h
                if overlap_area > face_area * 0.5:  # 50% overlap = duplicate
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_faces.append((x, y, w, h))
        
        # Remove small faces (likely false positives)
        min_face_area = gray.shape[0] * gray.shape[1] * 0.02  # At least 2% of image
        valid_faces = [(x, y, w, h) for (x, y, w, h) in all_faces if w * h >= min_face_area]
        
        # Validate faces by checking for eyes
        confirmed_faces = []
        for (x, y, w, h) in valid_faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            # A valid face should have at least 1 eye detected (allowing for partial view)
            if len(eyes) >= 1:
                confirmed_faces.append((x, y, w, h))
            else:
                # Still include if face is large enough (might be glasses blocking eye detection)
                if w * h >= min_face_area * 2:
                    confirmed_faces.append((x, y, w, h))
        
        num_faces = len(confirmed_faces)
        print(f"Faces detected: {num_faces} (raw: {len(all_faces)}, validated: {len(valid_faces)})")
        
        if num_faces == 0:
            return "NO_FACE", []
        elif num_faces == 1:
            return "ONE_FACE", confirmed_faces
        else:
            return "MULTIPLE_FACES", confirmed_faces
            
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return "ERROR", []

def extract_face_features(image, face_coords):
    """
    Extract comprehensive face features for comparison
    Uses multiple feature types for robust matching
    """
    try:
        x, y, w, h = face_coords
        
        # Add padding around face
        padding = int(min(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        # Resize to standard size for consistent comparison
        standard_size = (128, 128)
        face_resized = cv2.resize(face_region, standard_size)
        
        # Convert to different color spaces
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        hsv_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
        
        # Normalize lighting
        gray_face = cv2.equalizeHist(gray_face)
        
        # Apply slight blur to reduce noise
        gray_face = cv2.GaussianBlur(gray_face, (3, 3), 0)
        
        features = {}
        
        # 1. Grayscale histogram (256 bins)
        hist_gray = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        cv2.normalize(hist_gray, hist_gray, 0, 1, cv2.NORM_MINMAX)
        features['hist_gray'] = hist_gray.flatten()
        
        # 2. Color histograms (HSV)
        hist_h = cv2.calcHist([hsv_face], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_face], [1], None, [256], [0, 256])
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        features['hist_hue'] = hist_h.flatten()
        features['hist_saturation'] = hist_s.flatten()
        
        # 3. Local Binary Pattern (LBP) - texture features
        lbp = compute_lbp(gray_face)
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        cv2.normalize(hist_lbp, hist_lbp, 0, 1, cv2.NORM_MINMAX)
        features['hist_lbp'] = hist_lbp.flatten()
        
        # 4. Edge features using Sobel
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_direction = np.arctan2(sobely, sobelx)
        
        hist_edge_mag = cv2.calcHist([edge_magnitude.astype(np.float32)], [0], None, [64], [0, 255])
        cv2.normalize(hist_edge_mag, hist_edge_mag, 0, 1, cv2.NORM_MINMAX)
        features['hist_edge_magnitude'] = hist_edge_mag.flatten()
        
        # 5. Divide face into regions and compute regional features
        regions_features = []
        region_size = standard_size[0] // 4
        for i in range(4):
            for j in range(4):
                region = gray_face[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size]
                regions_features.append(np.mean(region))
                regions_features.append(np.std(region))
        features['regional_stats'] = np.array(regions_features)
        
        # 6. Structural features
        features['mean_intensity'] = np.mean(gray_face)
        features['std_intensity'] = np.std(gray_face)
        features['face_aspect_ratio'] = w / h
        
        # 7. HOG-like features (simplified)
        hog_features = compute_simple_hog(gray_face)
        features['hog'] = hog_features
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_lbp(gray_image):
    """Compute Local Binary Pattern"""
    rows, cols = gray_image.shape
    lbp = np.zeros_like(gray_image)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = gray_image[i, j]
            code = 0
            code |= (gray_image[i-1, j-1] >= center) << 7
            code |= (gray_image[i-1, j] >= center) << 6
            code |= (gray_image[i-1, j+1] >= center) << 5
            code |= (gray_image[i, j+1] >= center) << 4
            code |= (gray_image[i+1, j+1] >= center) << 3
            code |= (gray_image[i+1, j] >= center) << 2
            code |= (gray_image[i+1, j-1] >= center) << 1
            code |= (gray_image[i, j-1] >= center) << 0
            lbp[i, j] = code
    
    return lbp

def compute_simple_hog(gray_image):
    """Compute simplified HOG features"""
    # Compute gradients
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Create histogram of oriented gradients
    num_bins = 9
    bin_width = 180 // num_bins
    
    # Divide image into cells
    cell_size = 16
    cells_x = gray_image.shape[1] // cell_size
    cells_y = gray_image.shape[0] // cell_size
    
    hog_features = []
    for cy in range(cells_y):
        for cx in range(cells_x):
            cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            cell_dir = direction[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            
            hist = np.zeros(num_bins)
            for i in range(cell_size):
                for j in range(cell_size):
                    bin_idx = int(cell_dir[i, j] // bin_width) % num_bins
                    hist[bin_idx] += cell_mag[i, j]
            
            # Normalize
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            
            hog_features.extend(hist)
    
    return np.array(hog_features)

def compare_faces(features1, features2):
    """
    Compare two face feature sets
    Returns similarity score between 0 and 1
    """
    try:
        if features1 is None or features2 is None:
            return 0.0
        
        similarities = []
        weights = []
        
        # 1. Compare grayscale histograms (weight: 0.15)
        if 'hist_gray' in features1 and 'hist_gray' in features2:
            sim = cv2.compareHist(
                features1['hist_gray'].reshape(-1, 1).astype(np.float32),
                features2['hist_gray'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, sim))
            weights.append(0.15)
        
        # 2. Compare LBP histograms (weight: 0.25) - Most important for face recognition
        if 'hist_lbp' in features1 and 'hist_lbp' in features2:
            sim = cv2.compareHist(
                features1['hist_lbp'].reshape(-1, 1).astype(np.float32),
                features2['hist_lbp'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, sim))
            weights.append(0.25)
        
        # 3. Compare HOG features (weight: 0.25)
        if 'hog' in features1 and 'hog' in features2:
            hog1 = features1['hog']
            hog2 = features2['hog']
            if len(hog1) == len(hog2) and len(hog1) > 0:
                # Cosine similarity
                dot_product = np.dot(hog1, hog2)
                norm1 = np.linalg.norm(hog1)
                norm2 = np.linalg.norm(hog2)
                if norm1 > 0 and norm2 > 0:
                    sim = dot_product / (norm1 * norm2)
                    similarities.append(max(0, sim))
                    weights.append(0.25)
        
        # 4. Compare regional statistics (weight: 0.15)
        if 'regional_stats' in features1 and 'regional_stats' in features2:
            reg1 = features1['regional_stats']
            reg2 = features2['regional_stats']
            if len(reg1) == len(reg2):
                # Normalized correlation
                reg1_norm = (reg1 - np.mean(reg1)) / (np.std(reg1) + 1e-6)
                reg2_norm = (reg2 - np.mean(reg2)) / (np.std(reg2) + 1e-6)
                sim = np.corrcoef(reg1_norm, reg2_norm)[0, 1]
                if not np.isnan(sim):
                    similarities.append(max(0, sim))
                    weights.append(0.15)
        
        # 5. Compare edge magnitude histograms (weight: 0.10)
        if 'hist_edge_magnitude' in features1 and 'hist_edge_magnitude' in features2:
            sim = cv2.compareHist(
                features1['hist_edge_magnitude'].reshape(-1, 1).astype(np.float32),
                features2['hist_edge_magnitude'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, sim))
            weights.append(0.10)
        
        # 6. Compare hue histogram (weight: 0.05) - Skin tone
        if 'hist_hue' in features1 and 'hist_hue' in features2:
            sim = cv2.compareHist(
                features1['hist_hue'].reshape(-1, 1).astype(np.float32),
                features2['hist_hue'].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, sim))
            weights.append(0.05)
        
        # 7. Compare structural features (weight: 0.05)
        if all(k in features1 and k in features2 for k in ['mean_intensity', 'std_intensity']):
            mean_diff = abs(features1['mean_intensity'] - features2['mean_intensity'])
            std_diff = abs(features1['std_intensity'] - features2['std_intensity'])
            struct_sim = 1.0 - min((mean_diff + std_diff) / 100, 1.0)
            similarities.append(max(0, struct_sim))
            weights.append(0.05)
        
        # Calculate weighted average
        if len(similarities) > 0:
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(similarities, weights))
            final_similarity = weighted_sum / total_weight
            
            # Apply non-linear transformation to make differences more pronounced
            # This helps distinguish between same person (high similarity) and different people
            final_similarity = final_similarity ** 1.5  # Power transformation
            
            return max(0, min(1, final_similarity))
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error comparing faces: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def get_voter_hash(voter_id):
    """Get normalized voter ID hash for consistent storage"""
    if voter_id:
        return voter_id.lower().strip()
    return None

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    voter_hash = get_voter_hash(CURRENT_VOTER_ID)
    is_registered = voter_hash in REGISTERED_FACES if voter_hash else False
    
    return jsonify({
        'status': 'running',
        'face_registered': is_registered,
        'voter_id': CURRENT_VOTER_ID,
        'total_registered_voters': len(REGISTERED_FACES),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/check-registration', methods=['POST', 'OPTIONS'])
def check_registration():
    """Check if a voter already has a registered face"""
    global CURRENT_VOTER_ID
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        voter_id = data.get('voterId')
        
        if not voter_id:
            return jsonify({'error': 'No voter ID provided'}), 400
        
        voter_hash = get_voter_hash(voter_id)
        CURRENT_VOTER_ID = voter_id
        
        is_registered = voter_hash in REGISTERED_FACES
        
        return jsonify({
            'registered': is_registered,
            'voterId': voter_id,
            'message': 'Face already registered' if is_registered else 'Face not registered'
        })
        
    except Exception as e:
        print(f"Check registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/register-face', methods=['POST', 'OPTIONS'])
def register_face():
    """Register the authorized face for a specific voter account"""
    global REGISTERED_FACES, CURRENT_VOTER_ID
    
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        voter_id = data.get('voterId')
        image_base64 = data.get('image')
        force_reregister = data.get('forceReregister', False)
        
        print(f"\n{'='*60}")
        print(f"FACE REGISTRATION REQUEST")
        print(f"Voter ID: {voter_id}")
        print(f"Force Re-register: {force_reregister}")
        print(f"{'='*60}")
        
        if not voter_id:
            return jsonify({'success': False, 'error': 'No voter ID provided'}), 400
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        voter_hash = get_voter_hash(voter_id)
        CURRENT_VOTER_ID = voter_id
        
        # Check if already registered
        if voter_hash in REGISTERED_FACES and not force_reregister:
            return jsonify({
                'success': False,
                'error': 'Face already registered for this account',
                'alreadyRegistered': True,
                'message': 'This wallet already has a registered face. Use force re-register to update.'
            }), 400
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        print(f"Image size: {image.shape}")
        
        # Detect faces
        status, faces = detect_faces_robust(image)
        print(f"Face detection status: {status}, count: {len(faces)}")
        
        if status == "NO_FACE":
            return jsonify({
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible and well-lit.'
            }), 400
        
        elif status == "MULTIPLE_FACES":
            return jsonify({
                'success': False,
                'error': f'Multiple faces detected ({len(faces)}). Only one person should be in frame for registration.'
            }), 400
        
        elif status == "ONE_FACE":
            face_coords = faces[0]
            
            # Extract features
            face_features = extract_face_features(image, face_coords)
            if face_features is None:
                return jsonify({'success': False, 'error': 'Failed to extract facial features. Please try again.'}), 400
            
            # Store registration with voter ID
            REGISTERED_FACES[voter_hash] = {
                'features': face_features,
                'registered_at': datetime.now().isoformat(),
                'face_coords': face_coords,
                'voter_id': voter_id
            }
            
            x, y, w, h = face_coords
            print(f"✓ Face registered successfully for voter: {voter_id}")
            print(f"  Face area: {w}x{h}")
            print(f"  Total registered voters: {len(REGISTERED_FACES)}")
            
            return jsonify({
                'success': True,
                'message': 'Face registered successfully!',
                'voterId': voter_id,
                'faceCoords': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            })
        
        else:
            return jsonify({'success': False, 'error': 'Face detection error. Please try again.'}), 500
    
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/verify-face', methods=['POST', 'OPTIONS'])
def verify_face():
    """Verify if the current face matches the registered face for the voter"""
    global CURRENT_VOTER_ID
    
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        image_base64 = data.get('image')
        voter_id = data.get('voterId', CURRENT_VOTER_ID)
        
        if not voter_id:
            return jsonify({
                'verified': False,
                'reason': 'NO_VOTER_ID',
                'message': 'No voter ID provided',
                'similarity': 0.0
            })
        
        voter_hash = get_voter_hash(voter_id)
        
        # Check if this voter has registered
        if voter_hash not in REGISTERED_FACES:
            return jsonify({
                'verified': False,
                'reason': 'NOT_REGISTERED',
                'message': 'No face registered for this account. Please register first.',
                'similarity': 0.0
            })
        
        if not image_base64:
            return jsonify({
                'verified': False,
                'reason': 'NO_IMAGE',
                'message': 'No image provided',
                'similarity': 0.0
            })
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({
                'verified': False,
                'reason': 'INVALID_IMAGE',
                'message': 'Invalid image data',
                'similarity': 0.0
            })
        
        # Detect faces
        status, faces = detect_faces_robust(image)
        
        if status == "NO_FACE":
            return jsonify({
                'verified': False,
                'reason': 'NO_FACE',
                'message': 'No face detected. Please face the camera.',
                'similarity': 0.0
            })
        
        elif status == "MULTIPLE_FACES":
            return jsonify({
                'verified': False,
                'reason': 'MULTIPLE_FACES',
                'message': f'Multiple faces detected ({len(faces)}). Only the registered voter should be visible.',
                'similarity': 0.0
            })
        
        elif status == "ONE_FACE":
            face_coords = faces[0]
            
            # Extract current face features
            current_features = extract_face_features(image, face_coords)
            if current_features is None:
                return jsonify({
                    'verified': False,
                    'reason': 'FEATURE_ERROR',
                    'message': 'Failed to process face. Please try again.',
                    'similarity': 0.0
                })
            
            # Get registered features for this voter
            registered_data = REGISTERED_FACES[voter_hash]
            registered_features = registered_data['features']
            
            # Compare faces
            similarity = compare_faces(registered_features, current_features)
            
            print(f"Voter: {voter_id[:10]}... | Similarity: {similarity:.3f}")
            
            # Threshold for matching - STRICTER threshold
            MATCH_THRESHOLD = 0.55  # 55% similarity required
            
            if similarity >= MATCH_THRESHOLD:
                return jsonify({
                    'verified': True,
                    'similarity': float(similarity),
                    'message': f'Face verified ({similarity:.0%} match)'
                })
            else:
                return jsonify({
                    'verified': False,
                    'reason': 'FACE_MISMATCH',
                    'similarity': float(similarity),
                    'message': f'Face does not match registered face ({similarity:.0%}). Need {MATCH_THRESHOLD:.0%}.',
                    'threshold': MATCH_THRESHOLD
                })
        
        else:
            return jsonify({
                'verified': False,
                'reason': 'DETECTION_ERROR',
                'message': 'Face detection error',
                'similarity': 0.0
            })
    
    except Exception as e:
        print(f"Verification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'verified': False,
            'reason': 'SERVER_ERROR',
            'message': f'Server error: {str(e)}',
            'similarity': 0.0
        })

@app.route('/clear-registration', methods=['POST', 'OPTIONS'])
def clear_registration():
    """Clear the registered face for a specific voter"""
    global REGISTERED_FACES, CURRENT_VOTER_ID
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        voter_id = data.get('voterId', CURRENT_VOTER_ID)
        clear_all = data.get('clearAll', False)
        
        if clear_all:
            count = len(REGISTERED_FACES)
            REGISTERED_FACES = {}
            CURRENT_VOTER_ID = None
            print(f"✓ Cleared all {count} registrations")
            return jsonify({
                'success': True,
                'message': f'Cleared all {count} registrations'
            })
        
        if voter_id:
            voter_hash = get_voter_hash(voter_id)
            if voter_hash in REGISTERED_FACES:
                del REGISTERED_FACES[voter_hash]
                print(f"✓ Cleared registration for voter: {voter_id}")
                return jsonify({
                    'success': True,
                    'message': f'Registration cleared for {voter_id}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No registration found for this voter'
                })
        
        CURRENT_VOTER_ID = None
        return jsonify({
            'success': True,
            'message': 'Session cleared'
        })
        
    except Exception as e:
        print(f"Clear error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/set-voter', methods=['POST', 'OPTIONS'])
def set_voter():
    """Set the current voter ID for the session"""
    global CURRENT_VOTER_ID
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        voter_id = data.get('voterId')
        
        if not voter_id:
            return jsonify({'error': 'No voter ID provided'}), 400
        
        CURRENT_VOTER_ID = voter_id
        voter_hash = get_voter_hash(voter_id)
        is_registered = voter_hash in REGISTERED_FACES
        
        print(f"Session voter set: {voter_id} (registered: {is_registered})")
        
        return jsonify({
            'success': True,
            'voterId': voter_id,
            'isRegistered': is_registered
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['GET', 'OPTIONS'])
def debug():
    """Debug endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'current_voter': CURRENT_VOTER_ID,
        'total_registered': len(REGISTERED_FACES),
        'registered_voters': [
            {
                'voter_id': data['voter_id'][:10] + '...' if len(data['voter_id']) > 10 else data['voter_id'],
                'registered_at': data['registered_at']
            }
            for data in REGISTERED_FACES.values()
        ],
        'detectors_loaded': {
            'default': not face_cascade.empty(),
            'alt': not face_cascade_alt.empty(),
            'alt2': not face_cascade_alt2.empty(),
            'eye': not eye_cascade.empty()
        }
    })

@app.route('/stats', methods=['GET', 'OPTIONS'])
def stats():
    """Get registration statistics"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'total_registered_voters': len(REGISTERED_FACES),
        'current_session_voter': CURRENT_VOTER_ID,
        'server_uptime': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("IMPROVED Face Monitoring Backend")
    print("=" * 60)
    print(f"Server: http://localhost:5000")
    print(f"Match Threshold: 55%")
    print(f"Features: Per-account registration, LBP, HOG, Multi-cascade")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /register-face    - Register face for voter")
    print("  POST /verify-face      - Verify face matches registered")
    print("  POST /check-registration - Check if voter is registered")
    print("  POST /set-voter        - Set current session voter")
    print("  POST /clear-registration - Clear registration")
    print("  GET  /health           - Health check")
    print("  GET  /debug            - Debug info")
    print("  GET  /stats            - Statistics")
    print("=" * 60)
    
    app.run(debug=True, port=5000, threaded=True)