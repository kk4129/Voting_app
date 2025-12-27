# Robust Face Monitoring Backend v2.0 - OpenCV Only Version
# Advanced face matching without face_recognition dependency

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import os
import pickle
import hashlib
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

# Storage directory for persistent face data
FACE_DATA_DIR = 'face_database'
if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

# Global CORS handler
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# In-memory cache for faster access
FACE_CACHE = {}

# Initialize face detectors and recognizers
try:
    # Multiple Haar Cascades for robust detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    face_cascade_alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Initialize LBPH Face Recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=16,
        grid_x=8,
        grid_y=8,
        threshold=80.0
    )
    
    # Initialize ORB detector for keypoint features
    orb = cv2.ORB_create(nfeatures=500)
    
    # Initialize SIFT if available (better than ORB but patented)
    try:
        sift = cv2.SIFT_create(nfeatures=128)
        SIFT_AVAILABLE = True
    except:
        SIFT_AVAILABLE = False
    
    print("✓ Face detectors and recognizers loaded successfully")
    print(f"  SIFT available: {SIFT_AVAILABLE}")
    
except Exception as e:
    print(f"Error loading face detectors: {e}")

def get_account_hash(voter_id):
    """Generate a safe filename from voter ID"""
    if not voter_id:
        return None
    return hashlib.sha256(voter_id.lower().strip().encode()).hexdigest()[:16]

def save_face_data(voter_id, face_data):
    """Persist face data to disk"""
    account_hash = get_account_hash(voter_id)
    if not account_hash:
        return False
    
    filepath = os.path.join(FACE_DATA_DIR, f"{account_hash}.pkl")
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(face_data, f)
        
        FACE_CACHE[voter_id.lower().strip()] = face_data
        print(f"✓ Face data saved for {voter_id[:10]}...")
        return True
    except Exception as e:
        print(f"Error saving face data: {e}")
        return False

def load_face_data(voter_id):
    """Load face data from disk"""
    voter_key = voter_id.lower().strip()
    
    # Check cache first
    if voter_key in FACE_CACHE:
        return FACE_CACHE[voter_key]
    
    account_hash = get_account_hash(voter_id)
    if not account_hash:
        return None
    
    filepath = os.path.join(FACE_DATA_DIR, f"{account_hash}.pkl")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            face_data = pickle.load(f)
        
        FACE_CACHE[voter_key] = face_data
        print(f"✓ Face data loaded for {voter_id[:10]}...")
        return face_data
    except Exception as e:
        print(f"Error loading face data: {e}")
        return None

def delete_face_data(voter_id):
    """Delete face data from disk and cache"""
    voter_key = voter_id.lower().strip()
    
    if voter_key in FACE_CACHE:
        del FACE_CACHE[voter_key]
    
    account_hash = get_account_hash(voter_id)
    if account_hash:
        filepath = os.path.join(FACE_DATA_DIR, f"{account_hash}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Face data deleted for {voter_id[:10]}...")
            return True
    
    return False

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

def preprocess_face(face_image):
    """Preprocess face image for better recognition"""
    # Convert to grayscale
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    # Resize to standard size
    standard_size = (150, 150)
    resized = cv2.resize(gray, standard_size)
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(resized)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    return filtered

def detect_faces_multi_cascade(image):
    """
    Robust face detection using multiple cascades and validation
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        all_detections = []
        
        # Parameters for each cascade
        cascade_params = [
            (face_cascade, 1.1, 5, (50, 50)),
            (face_cascade_alt, 1.1, 4, (50, 50)),
            (face_cascade_alt2, 1.15, 5, (60, 60)),
            (face_cascade_alt_tree, 1.1, 5, (50, 50))
        ]
        
        # Detect with multiple cascades
        for cascade, scale, neighbors, min_size in cascade_params:
            faces = cascade.detectMultiScale(
                gray_eq,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                all_detections.append((x, y, w, h))
        
        # Merge overlapping detections using Non-Maximum Suppression
        merged_faces = non_maximum_suppression(all_detections, overlap_thresh=0.3)
        
        # Validate faces with eye detection and aspect ratio
        validated_faces = []
        for (x, y, w, h) in merged_faces:
            # Check aspect ratio (faces are typically 1:1.2 to 1:1.5)
            aspect_ratio = h / w
            if aspect_ratio < 0.8 or aspect_ratio > 2.0:
                continue
            
            # Check for minimum size (at least 3% of image)
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            if face_area < image_area * 0.03:
                continue
            
            # Validate with eye detection
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            # Accept face if at least 1 eye detected or face is large enough
            if len(eyes) >= 1 or face_area > image_area * 0.1:
                validated_faces.append((x, y, w, h))
        
        num_faces = len(validated_faces)
        print(f"Detection: {len(all_detections)} raw, {len(merged_faces)} merged, {num_faces} validated")
        
        if num_faces == 0:
            return "NO_FACE", []
        elif num_faces == 1:
            return "ONE_FACE", validated_faces
        else:
            return "MULTIPLE_FACES", validated_faces
            
    except Exception as e:
        print(f"Error in face detection: {e}")
        return "ERROR", []

def non_maximum_suppression(boxes, overlap_thresh=0.3):
    """Apply Non-Maximum Suppression to merge overlapping detections"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    
    # Compute area of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Sort by y2 (bottom of box)
    idxs = np.argsort(y2)
    
    picked = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)
        
        # Find overlap with all other boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[idxs[:last]]
        
        # Delete boxes with high overlap
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return boxes[picked].tolist()

def extract_face_embeddings(image, face_coords):
    """
    Extract comprehensive face embeddings using multiple techniques
    """
    try:
        x, y, w, h = face_coords
        
        # Add padding
        padding = int(min(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        # Preprocess face
        face_processed = preprocess_face(face_region)
        
        embeddings = {}
        
        # 1. LBPH (Local Binary Pattern Histogram)
        lbph_hist = compute_lbph_histogram(face_processed)
        embeddings['lbph'] = lbph_hist
        
        # 2. HOG (Histogram of Oriented Gradients)
        hog_features = compute_hog_features(face_processed)
        embeddings['hog'] = hog_features
        
        # 3. Gabor Wavelets
        gabor_features = compute_gabor_features(face_processed)
        embeddings['gabor'] = gabor_features
        
        # 4. Keypoint descriptors (SIFT or ORB)
        if SIFT_AVAILABLE:
            keypoints, descriptors = sift.detectAndCompute(face_processed, None)
            if descriptors is not None and len(descriptors) > 0:
                # Take mean of descriptors as feature
                embeddings['sift'] = np.mean(descriptors, axis=0)
            else:
                embeddings['sift'] = np.zeros(128)
        else:
            keypoints, descriptors = orb.detectAndCompute(face_processed, None)
            if descriptors is not None and len(descriptors) > 0:
                embeddings['orb'] = np.mean(descriptors.astype(float), axis=0)
            else:
                embeddings['orb'] = np.zeros(32)
        
        # 5. Statistical features
        embeddings['statistical'] = compute_statistical_features(face_processed)
        
        # 6. Face geometry (landmarks approximation using edges)
        embeddings['geometry'] = compute_geometric_features(face_processed)
        
        # 7. Color features (if color image)
        if len(image.shape) == 3:
            embeddings['color'] = compute_color_features(face_region)
        
        # Store the processed face for LBPH recognizer
        embeddings['processed_face'] = face_processed
        
        return embeddings
        
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_lbph_histogram(face_gray):
    """Compute Local Binary Pattern Histogram"""
    # Divide image into cells
    cell_size = 16
    h, w = face_gray.shape
    cells_x = w // cell_size
    cells_y = h // cell_size
    
    histogram = []
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            cell = face_gray[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            
            # Compute LBP for cell
            lbp_image = np.zeros_like(cell)
            for i in range(1, cell.shape[0]-1):
                for j in range(1, cell.shape[1]-1):
                    center = cell[i, j]
                    code = 0
                    code |= (cell[i-1, j-1] >= center) << 7
                    code |= (cell[i-1, j] >= center) << 6
                    code |= (cell[i-1, j+1] >= center) << 5
                    code |= (cell[i, j+1] >= center) << 4
                    code |= (cell[i+1, j+1] >= center) << 3
                    code |= (cell[i+1, j] >= center) << 2
                    code |= (cell[i+1, j-1] >= center) << 1
                    code |= (cell[i, j-1] >= center) << 0
                    lbp_image[i, j] = code
            
            # Compute histogram for cell
            hist, _ = np.histogram(lbp_image, bins=59, range=(0, 256))
            histogram.extend(hist)
    
    histogram = np.array(histogram, dtype=np.float32)
    # Normalize
    histogram = histogram / (np.sum(histogram) + 1e-7)
    
    return histogram

def compute_hog_features(face_gray):
    """Compute Histogram of Oriented Gradients features"""
    # Compute gradients
    gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Create histogram
    cell_size = 16
    num_bins = 9
    bin_width = 180 / num_bins
    
    h, w = face_gray.shape
    cells_x = w // cell_size
    cells_y = h // cell_size
    
    hog_features = []
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            cell_dir = direction[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            
            hist = np.zeros(num_bins)
            for i in range(cell_size):
                for j in range(cell_size):
                    if i < cell_mag.shape[0] and j < cell_mag.shape[1]:
                        bin_idx = int(cell_dir[i, j] / bin_width) % num_bins
                        hist[bin_idx] += cell_mag[i, j]
            
            # Normalize
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            
            hog_features.extend(hist)
    
    return np.array(hog_features, dtype=np.float32)

def compute_gabor_features(face_gray):
    """Compute Gabor wavelet features"""
    features = []
    
    # Gabor parameters
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    
    # Multiple orientations
    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
        filtered = cv2.filter2D(face_gray, cv2.CV_8UC3, kernel)
        
        # Extract statistics from filtered image
        features.extend([
            np.mean(filtered),
            np.std(filtered),
            np.percentile(filtered, 25),
            np.percentile(filtered, 75)
        ])
    
    return np.array(features, dtype=np.float32)

def compute_statistical_features(face_gray):
    """Compute statistical features from face"""
    features = []
    
    # Global statistics
    features.extend([
        np.mean(face_gray),
        np.std(face_gray),
        np.median(face_gray),
        np.percentile(face_gray, 25),
        np.percentile(face_gray, 75),
        np.max(face_gray) - np.min(face_gray)  # Range
    ])
    
    # Divide into regions and compute local statistics
    h, w = face_gray.shape
    regions = [
        face_gray[0:h//2, 0:w//2],      # Top-left
        face_gray[0:h//2, w//2:w],      # Top-right
        face_gray[h//2:h, 0:w//2],      # Bottom-left
        face_gray[h//2:h, w//2:w],      # Bottom-right
        face_gray[h//3:2*h//3, w//3:2*w//3]  # Center
    ]
    
    for region in regions:
        features.extend([
            np.mean(region),
            np.std(region),
            np.median(region)
        ])
    
    return np.array(features, dtype=np.float32)

def compute_geometric_features(face_gray):
    """Compute geometric features using edge detection"""
    # Detect edges
    edges = cv2.Canny(face_gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    if contours:
        # Find largest contour (likely face outline)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Hu moments (shape descriptors)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        features.extend([
            area / (face_gray.shape[0] * face_gray.shape[1]),  # Relative area
            perimeter / (2 * (face_gray.shape[0] + face_gray.shape[1])),  # Relative perimeter
            4 * np.pi * area / (perimeter ** 2 + 1e-7)  # Circularity
        ])
        features.extend(hu_moments[:4])  # First 4 Hu moments
    else:
        features.extend([0] * 7)
    
    return np.array(features, dtype=np.float32)

def compute_color_features(face_color):
    """Compute color-based features"""
    features = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
    
    # HSV statistics (mainly for skin tone)
    h, s, v = cv2.split(hsv)
    features.extend([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ])
    
    # LAB statistics
    l, a, b = cv2.split(lab)
    features.extend([
        np.mean(l), np.std(l),
        np.mean(a), np.std(a),
        np.mean(b), np.std(b)
    ])
    
    return np.array(features, dtype=np.float32)

def compare_face_embeddings(embeddings1, embeddings2):
    """
    Compare two face embeddings using multiple similarity metrics
    """
    try:
        if not embeddings1 or not embeddings2:
            return 0.0
        
        similarities = []
        weights = []
        
        # 1. LBPH comparison (most important for face recognition)
        if 'lbph' in embeddings1 and 'lbph' in embeddings2:
            lbph_sim = 1.0 - min(1.0, distance.euclidean(embeddings1['lbph'], embeddings2['lbph']) / 10)
            similarities.append(lbph_sim)
            weights.append(0.30)
        
        # 2. HOG comparison
        if 'hog' in embeddings1 and 'hog' in embeddings2:
            hog_sim = 1.0 - distance.cosine(embeddings1['hog'], embeddings2['hog'])
            similarities.append(max(0, hog_sim))
            weights.append(0.25)
        
        # 3. Gabor comparison
        if 'gabor' in embeddings1 and 'gabor' in embeddings2:
            gabor_sim = 1.0 - distance.cosine(embeddings1['gabor'], embeddings2['gabor'])
            similarities.append(max(0, gabor_sim))
            weights.append(0.15)
        
        # 4. SIFT/ORB comparison
        if 'sift' in embeddings1 and 'sift' in embeddings2:
            sift_sim = 1.0 - distance.cosine(embeddings1['sift'], embeddings2['sift'])
            similarities.append(max(0, sift_sim))
            weights.append(0.20)
        elif 'orb' in embeddings1 and 'orb' in embeddings2:
            orb_sim = 1.0 - distance.cosine(embeddings1['orb'], embeddings2['orb'])
            similarities.append(max(0, orb_sim))
            weights.append(0.15)
        
        # 5. Statistical features
        if 'statistical' in embeddings1 and 'statistical' in embeddings2:
            stat_sim = 1.0 - min(1.0, distance.euclidean(
                embeddings1['statistical'] / (np.max(embeddings1['statistical']) + 1e-7),
                embeddings2['statistical'] / (np.max(embeddings2['statistical']) + 1e-7)
            ))
            similarities.append(stat_sim)
            weights.append(0.05)
        
        # 6. Geometric features
        if 'geometry' in embeddings1 and 'geometry' in embeddings2:
            geo_sim = 1.0 - distance.cosine(embeddings1['geometry'], embeddings2['geometry'])
            similarities.append(max(0, geo_sim))
            weights.append(0.05)
        
        # 7. Color features
        if 'color' in embeddings1 and 'color' in embeddings2:
            color_sim = 1.0 - distance.cosine(embeddings1['color'], embeddings2['color'])
            similarities.append(max(0, color_sim))
            weights.append(0.05)
        
        # 8. Use LBPH recognizer if faces available
        if 'processed_face' in embeddings1 and 'processed_face' in embeddings2:
            try:
                # Train recognizer on registered face
                face_recognizer.train([embeddings1['processed_face']], np.array([0]))
                
                # Predict on current face
                label, confidence = face_recognizer.predict(embeddings2['processed_face'])
                
                # Convert confidence to similarity (lower confidence = higher similarity)
                # LBPH confidence typically ranges from 0 to 100
                lbph_recognizer_sim = max(0, 1 - (confidence / 100))
                similarities.append(lbph_recognizer_sim)
                weights.append(0.35)  # High weight for dedicated recognizer
            except:
                pass
        
        # Calculate weighted average
        if len(similarities) > 0:
            # Normalize weights
            total_weight = sum(weights[:len(similarities)])
            normalized_weights = [w / total_weight for w in weights[:len(similarities)]]
            
            # Weighted average
            final_similarity = sum(s * w for s, w in zip(similarities, normalized_weights))
            
            # Apply non-linear transformation for better discrimination
            # This makes the difference between same/different faces more pronounced
            if final_similarity > 0.6:
                final_similarity = 0.6 + (final_similarity - 0.6) * 1.5
            else:
                final_similarity = final_similarity * 0.9
            
            final_similarity = max(0, min(1, final_similarity))
            
            print(f"Similarities - LBPH: {similarities[0]:.3f}, HOG: {similarities[1] if len(similarities) > 1 else 0:.3f}, Final: {final_similarity:.3f}")
            
            return final_similarity
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error comparing embeddings: {e}")
        return 0.0

# API Endpoints

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    registered_count = 0
    if os.path.exists(FACE_DATA_DIR):
        registered_count = len([f for f in os.listdir(FACE_DATA_DIR) if f.endswith('.pkl')])
    
    return jsonify({
        'status': 'running',
        'total_registered_voters': registered_count,
        'cache_size': len(FACE_CACHE),
        'opencv_version': cv2.__version__,
        'sift_available': SIFT_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/check-registration', methods=['POST', 'OPTIONS'])
def check_registration():
    """Check if a voter already has a registered face"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        voter_id = data.get('voterId')
        
        if not voter_id:
            return jsonify({'error': 'No voter ID provided'}), 400
        
        face_data = load_face_data(voter_id)
        is_registered = face_data is not None
        
        return jsonify({
            'registered': is_registered,
            'voterId': voter_id,
            'message': 'Face already registered' if is_registered else 'Face not registered',
            'registeredAt': face_data.get('registered_at') if face_data else None
        })
        
    except Exception as e:
        print(f"Check registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/register-face', methods=['POST', 'OPTIONS'])
def register_face():
    """Register the face for a specific voter account"""
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
        
        if not voter_id or not image_base64:
            return jsonify({
                'success': False,
                'error': 'Voter ID and image required'
            }), 400
        
        # Check existing registration
        existing_face = load_face_data(voter_id)
        if existing_face and not force_reregister:
            return jsonify({
                'success': False,
                'error': 'Face already registered for this account',
                'alreadyRegistered': True,
                'registeredAt': existing_face.get('registered_at')
            }), 400
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        print(f"Image shape: {image.shape}")
        
        # Detect faces
        status, faces = detect_faces_multi_cascade(image)
        
        if status == "NO_FACE":
            return jsonify({
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible.'
            }), 400
        
        elif status == "MULTIPLE_FACES":
            return jsonify({
                'success': False,
                'error': f'Multiple faces detected ({len(faces)}). Only one person should be in frame.'
            }), 400
        
        elif status == "ONE_FACE":
            face_coords = faces[0]
            
            # Extract face embeddings
            face_embeddings = extract_face_embeddings(image, face_coords)
            
            if face_embeddings is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract facial features. Please try again.'
                }), 400
            
            # Prepare data for storage
            face_data = {
                'voter_id': voter_id,
                'embeddings': face_embeddings,
                'face_coords': face_coords,
                'registered_at': datetime.now().isoformat(),
                'image_shape': image.shape
            }
            
            # Save to persistent storage
            if save_face_data(voter_id, face_data):
                print(f"✅ Face registered successfully for: {voter_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'Face registered successfully!',
                    'voterId': voter_id,
                    'registeredAt': face_data['registered_at']
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save face data'
                }), 500
        
        else:
            return jsonify({
                'success': False,
                'error': 'Face detection error'
            }), 500
    
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/verify-face', methods=['POST', 'OPTIONS'])
def verify_face():
    """Verify if the current face matches the registered face"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        image_base64 = data.get('image')
        voter_id = data.get('voterId')
        
        if not voter_id:
            return jsonify({
                'verified': False,
                'reason': 'NO_VOTER_ID',
                'message': 'No voter ID provided',
                'similarity': 0.0
            })
        
        # Load registered face data
        registered_face = load_face_data(voter_id)
        
        if not registered_face:
            return jsonify({
                'verified': False,
                'reason': 'NOT_REGISTERED',
                'message': 'No face registered for this account',
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
        status, faces = detect_faces_multi_cascade(image)
        
        if status == "NO_FACE":
            return jsonify({
                'verified': False,
                'reason': 'NO_FACE',
                'message': 'No face detected',
                'similarity': 0.0
            })
        
        elif status == "MULTIPLE_FACES":
            return jsonify({
                'verified': False,
                'reason': 'MULTIPLE_FACES',
                'message': f'{len(faces)} faces detected. Only one allowed.',
                'similarity': 0.0
            })
        
        elif status == "ONE_FACE":
            # Extract current face embeddings
            current_embeddings = extract_face_embeddings(image, faces[0])
            
            if current_embeddings is None:
                return jsonify({
                    'verified': False,
                    'reason': 'FEATURE_ERROR',
                    'message': 'Failed to process face',
                    'similarity': 0.0
                })
            
            # Compare with registered face
            registered_embeddings = registered_face.get('embeddings')
            similarity = compare_face_embeddings(registered_embeddings, current_embeddings)
            
            print(f"Verification for {voter_id[:10]}... | Similarity: {similarity:.3f}")
            
            # Threshold for matching
            MATCH_THRESHOLD = 0.60  # 60% similarity required
            
            if similarity >= MATCH_THRESHOLD:
                return jsonify({
                    'verified': True,
                    'similarity': float(similarity),
                    'message': f'Face verified ({similarity:.0%} match)',
                    'threshold': MATCH_THRESHOLD
                })
            else:
                return jsonify({
                    'verified': False,
                    'reason': 'FACE_MISMATCH',
                    'similarity': float(similarity),
                    'message': f'Face mismatch ({similarity:.0%} < {MATCH_THRESHOLD:.0%})',
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
    """Clear the registered face for a voter"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        voter_id = data.get('voterId')
        clear_all = data.get('clearAll', False)
        
        if clear_all:
            count = 0
            if os.path.exists(FACE_DATA_DIR):
                for file in os.listdir(FACE_DATA_DIR):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(FACE_DATA_DIR, file))
                        count += 1
            
            FACE_CACHE.clear()
            print(f"✓ Cleared all {count} registrations")
            
            return jsonify({
                'success': True,
                'message': f'Cleared all {count} registrations'
            })
        
        if voter_id:
            if delete_face_data(voter_id):
                return jsonify({
                    'success': True,
                    'message': f'Registration cleared for {voter_id}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No registration found'
                })
        
        return jsonify({
            'success': False,
            'message': 'No voter ID provided'
        })
        
    except Exception as e:
        print(f"Clear error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET', 'OPTIONS'])
def stats():
    """Get registration statistics"""
    if request.method == 'OPTIONS':
        return '', 204
    
    registered_count = 0
    if os.path.exists(FACE_DATA_DIR):
        registered_count = len([f for f in os.listdir(FACE_DATA_DIR) if f.endswith('.pkl')])
    
    return jsonify({
        'total_registered_voters': registered_count,
        'cache_size': len(FACE_CACHE),
        'storage_directory': FACE_DATA_DIR,
        'opencv_version': cv2.__version__,
        'sift_available': SIFT_AVAILABLE,
        'server_uptime': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ROBUST Face Monitoring Backend v2.0 (OpenCV Edition)")
    print("=" * 60)
    print(f"Features:")
    print(f"  • Advanced multi-cascade face detection")
    print(f"  • LBPH, HOG, Gabor feature extraction")
    print(f"  • Persistent storage in: {FACE_DATA_DIR}/")
    print(f"  • Per-account face registration")
    print(f"  • SIFT available: {SIFT_AVAILABLE}")
    print("=" * 60)
    print(f"Server: http://localhost:5000")
    print(f"Match Threshold: 60%")
    print("=" * 60)
    
    # Check dependencies
    try:
        from scipy import __version__ as scipy_version
        print(f"✓ SciPy {scipy_version}")
    except ImportError:
        print("⚠ SciPy not installed. Run: pip install scipy")
    
    try:
        from sklearn import __version__ as sklearn_version
        print(f"✓ Scikit-learn {sklearn_version}")
    except ImportError:
        print("⚠ Scikit-learn not installed. Run: pip install scikit-learn")
    
    print(f"✓ OpenCV {cv2.__version__}")
    print("=" * 60)
    
    app.run(debug=True, port=5000, threaded=True)