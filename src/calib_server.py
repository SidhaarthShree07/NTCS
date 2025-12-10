import os
import json
import requests
import supervision as sv
from collections import defaultdict, deque
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request
from flask_cors import CORS
import track   # 🔹 segmentation + polygon utils
import gemini_lines
import perspective_calibration  # 🔹 New perspective-aware calibration
import auto_homography  # 🔹 NEW: Automatic homography from road polygon + vehicles
import vehicle_size_verification  # 🔹 NEW: LLM-based vehicle size comparison for distance verification
import tracked_vehicle_calibration  # 🔹 NEW: Track single vehicle A→B for LLM-first calibration
import glob
from violation_recorder import create_violation_recorder  # 🔹 NEW: Violation recording system

# Set OpenCV environment variables for HLS support BEFORE any video operations
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'allowed_extensions;ALL'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'

# Backend API configuration
BACKEND_API_BASE = "https://nextgen-fv1h.onrender.com/api"
current_camera_id = None  # Track current camera being calibrated

# Paths
BASE_DIR = os.path.dirname(__file__)
CALIB_DIR = os.path.join(BASE_DIR, 'calib')
TEMP_VIDEO_DIR = os.path.join(BASE_DIR, 'temp')  # For cached HTTP streams
CACHE_METADATA_FILE = os.path.join(TEMP_VIDEO_DIR, 'cache_metadata.json')  # Cache tracking

# Video source configuration - will be set via API when camera is added
current_video_source = None  # No default video source
cap = None
frame_lock = threading.Lock()
current_frame = None
model_frame = None
model_lock = threading.Lock()

# Violation recorder instance
violation_recorder = None
violation_recorder_lock = threading.Lock()

# Global cleanup thread for temp files (runs independently of speed detection)
cleanup_thread_running = True
cleanup_thread = None

def start_global_cleanup_thread():
    """Start background cleanup thread that runs every 2 minutes."""
    global cleanup_thread, cleanup_thread_running
    
    if cleanup_thread is not None and cleanup_thread.is_alive():
        print("[GlobalCleanup] Cleanup thread already running")
        return
    
    def cleanup_loop():
        from pathlib import Path
        import time
        
        proof_dir = Path("temp/proof")
        proof_dir.mkdir(parents=True, exist_ok=True)
        
        print("[GlobalCleanup] 🧹 Cleanup thread started - runs every 2 minutes")
        iteration = 0
        
        while cleanup_thread_running:
            try:
                time.sleep(120)  # Run every 2 minutes
                iteration += 1
                print(f"[GlobalCleanup] 🔄 Running scheduled cleanup #{iteration} (every 2 min)")
                
                current_time = time.time()
                cleanup_threshold = 600  # 10 minutes
                deleted_count = 0
                total_files = 0
                
                for file_path in proof_dir.glob('*'):
                    if file_path.is_file():
                        total_files += 1
                        file_age = current_time - file_path.stat().st_mtime
                        
                        if file_age > cleanup_threshold:
                            try:
                                file_path.unlink()
                                deleted_count += 1
                                print(f"[GlobalCleanup]   🗑️ Deleted: {file_path.name} (age: {file_age/60:.1f}min)")
                            except Exception as e:
                                print(f"[GlobalCleanup] Failed to delete {file_path.name}: {e}")
                
                print(f"[GlobalCleanup] 🧹 Cleanup complete: {deleted_count}/{total_files} files deleted (threshold: 10min)")
                
            except Exception as e:
                print(f"[GlobalCleanup] Cleanup error: {e}")
    
    cleanup_thread_running = True
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("[GlobalCleanup] ✅ Global cleanup thread started")

# Video caching system for HTTP streams
video_cache = {
    'url': None,           # Current cached URL
    'file_path': None,     # Path to cached file
    'download_time': 0,    # Timestamp of last download
    'is_downloading': False,
    'download_progress': 0,  # 0-100%
    'is_ready': True,      # False during initial download
    'error': None
}
video_cache_lock = threading.Lock()
CACHE_REFRESH_INTERVAL = 60 * 60  # 60 minutes in seconds


app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ---------------------------
# Config
# ---------------------------

# Calibration store
calib = {
    'line_A_y': 300,
    'line_B_y': 500,
    'calib_distance_m': 10.0,
    'road_width_m': 10.0,  # Road width from Gemini
    'running': False,
    'source_points': None,
    'confidence_score': 0.0,
    'calibration_method': 'unknown',
    'homography_matrix': None,  # 🔹 NEW: 3x3 matrix for IPM (if calibrated)
    'homography_points': None   # 🔹 NEW: Store calibration points for reference
}

# YOLO class names mapping (COCO dataset)
CLASS_NAMES_DICT = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# 🔹 Load segmentation model once
seg_model = track.load_segmentation_model()


# ---------------------------
# Utils
# ---------------------------
def ensure_calib_dir_clean():
    """Create/clean the calib folder so only latest screenshots are kept."""
    os.makedirs(CALIB_DIR, exist_ok=True)
    # remove files inside
    for p in glob.glob(os.path.join(CALIB_DIR, '*')):
        try:
            os.remove(p)
        except Exception:
            pass


def ensure_temp_dir():
    """Create temp directory for cached videos."""
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)


def load_cache_metadata():
    """Load cache metadata from JSON file."""
    if not os.path.exists(CACHE_METADATA_FILE):
        return {}
    
    try:
        with open(CACHE_METADATA_FILE, 'r') as f:
            data = json.load(f)
            print(f"[CacheMetadata] Loaded {len(data)} cache entries")
            return data
    except Exception as e:
        print(f"[CacheMetadata] Failed to load: {e}")
        return {}


def save_cache_metadata(metadata):
    """Save cache metadata to JSON file."""
    try:
        ensure_temp_dir()
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[CacheMetadata] Saved {len(metadata)} cache entries")
    except Exception as e:
        print(f"[CacheMetadata] Failed to save: {e}")


def get_cache_entry(url):
    """Get cache metadata for a specific URL."""
    metadata = load_cache_metadata()
    return metadata.get(url)


def update_cache_entry(url, file_path, download_time):
    """Update cache metadata for a URL."""
    metadata = load_cache_metadata()
    metadata[url] = {
        'url': url,
        'file_path': file_path,
        'download_time': download_time,
        'last_accessed': time.time()
    }
    save_cache_metadata(metadata)


def cleanup_old_caches():
    """Remove cache files that are no longer in metadata (orphaned files)."""
    if not os.path.exists(TEMP_VIDEO_DIR):
        return
    
    metadata = load_cache_metadata()
    active_files = {entry['file_path'] for entry in metadata.values() if 'file_path' in entry}
    
    # Find all .mp4 files in temp dir
    for filename in os.listdir(TEMP_VIDEO_DIR):
        if filename.endswith('.mp4'):
            filepath = os.path.join(TEMP_VIDEO_DIR, filename)
            if filepath not in active_files:
                try:
                    print(f"[CacheCleanup] Removing orphaned file: {filename}")
                    os.remove(filepath)
                except Exception as e:
                    print(f"[CacheCleanup] Failed to remove {filename}: {e}")


# ---------------------------
# Backend API Integration
# ---------------------------

def fetch_calibration_data(camera_id):
    """
    Fetch calibration data from backend API.
    GET https://nextgen-fv1h.onrender.com/api/calibration/{camera_id}
    Returns calibration data or None if not found.
    """
    try:
        url = f"{BACKEND_API_BASE}/calibration/{camera_id}"
        print(f"[BackendAPI] Fetching calibration: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[BackendAPI] Calibration data received: {data}")
            return data
        elif response.status_code == 404:
            print(f"[BackendAPI] No calibration found for camera {camera_id}")
            return None
        else:
            print(f"[BackendAPI] Error fetching calibration: {response.status_code}")
            return None
    except Exception as e:
        print(f"[BackendAPI] Failed to fetch calibration: {e}")
        return None


def save_calibration_to_backend(camera_id, camera_link, location, stats):
    """
    Save calibration data to backend API.
    POST https://nextgen-fv1h.onrender.com/api/calibration
    """
    try:
        url = f"{BACKEND_API_BASE}/calibration"
        payload = {
            "cameraId": camera_id,
            "cameraLink": camera_link,
            "location": location,
            "currentStatus": "ACTIVE",
            "stats": stats
        }
        
        print(f"[BackendAPI] Saving calibration to: {url}")
        print(f"[BackendAPI] Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"[BackendAPI] Calibration saved successfully")
            return True
        else:
            print(f"[BackendAPI] Error saving calibration: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[BackendAPI] Failed to save calibration: {e}")
        return False


def fetch_speed_limit(camera_id):
    """
    Fetch speed limit for camera from backend API.
    GET https://nextgen-fv1h.onrender.com/api/speed-limit?cameraId={camera_id}
    Returns 75 km/h as fallback if API fails.
    """
    try:
        url = f"{BACKEND_API_BASE}/speed-limit"
        params = {"cameraId": camera_id}
        print(f"[BackendAPI] Fetching speed limit: {url}?cameraId={camera_id}")
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            speed_limit = data.get('speedLimit') or data.get('speed_limit')
            if speed_limit:
                print(f"[BackendAPI] Speed limit for {camera_id}: {speed_limit} km/h")
                return speed_limit
            else:
                print(f"[BackendAPI] No speed limit in response, using fallback: 75 km/h")
                return 75
        else:
            print(f"[BackendAPI] Error fetching speed limit: {response.status_code}, using fallback: 75 km/h")
            return 75
    except Exception as e:
        print(f"[BackendAPI] Failed to fetch speed limit: {e}, using fallback: 75 km/h")
        return 75


def update_camera_status(camera_id, status):
    """
    Update camera detection status (ACTIVE/INACTIVE).
    POST https://nextgen-fv1h.onrender.com/api/calibration/status/{camera_id}
    """
    try:
        url = f"{BACKEND_API_BASE}/calibration/status/{camera_id}"
        payload = {"status": status}
        
        print(f"[BackendAPI] Updating status: {url}")
        print(f"[BackendAPI] Status: {status}")
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"[BackendAPI] Status updated successfully")
            return True
        else:
            print(f"[BackendAPI] Error updating status: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[BackendAPI] Failed to update status: {e}")
        return False


def check_camera_status(camera_id):
    """
    Check if camera status is ACTIVE from backend.
    Returns True if ACTIVE, False otherwise.
    """
    try:
        # Fetch calibration data which includes currentStatus
        data = fetch_calibration_data(camera_id)
        if data and data.get('currentStatus') == 'ACTIVE':
            return True
        return False
    except Exception as e:
        print(f"[BackendAPI] Failed to check camera status: {e}")
        return False


def download_and_cache_stream(url, force_refresh=False):
    """
    Download HTTP stream to temp folder with smart caching.
    Returns path to cached file or None on error.
    Persists cache metadata to JSON file.
    """
    global video_cache
    
    with video_cache_lock:
        current_time = time.time()
        
        # Check metadata file first
        cache_entry = get_cache_entry(url)
        
        # Check if we already have a valid cache
        if (cache_entry and 
            cache_entry.get('file_path') and 
            os.path.exists(cache_entry['file_path']) and
            not force_refresh and
            (current_time - cache_entry.get('download_time', 0)) < CACHE_REFRESH_INTERVAL):
            print(f"[VideoCache] Using cached file from metadata: {cache_entry['file_path']}")
            print(f"[VideoCache] Cache age: {int((current_time - cache_entry['download_time']) / 60)} minutes")
            video_cache['url'] = url
            video_cache['file_path'] = cache_entry['file_path']
            video_cache['download_time'] = cache_entry['download_time']
            video_cache['is_ready'] = True
            # Update last accessed time
            update_cache_entry(url, cache_entry['file_path'], cache_entry['download_time'])
            return cache_entry['file_path']
        
        # Need to download
        if video_cache['is_downloading']:
            print(f"[VideoCache] Download already in progress...")
            # If we have an old cache file, return it while downloading
            if cache_entry and cache_entry.get('file_path') and os.path.exists(cache_entry['file_path']):
                print(f"[VideoCache] Using old cache while downloading: {cache_entry['file_path']}")
                return cache_entry['file_path']
            # Otherwise wait for download
            return None
        
        video_cache['is_downloading'] = True
        video_cache['download_progress'] = 0
        video_cache['is_ready'] = False
        video_cache['error'] = None
    
    # Download in background
    import requests
    import hashlib
    
    try:
        # Create temp directory
        ensure_temp_dir()
        
        # Generate unique filename based on URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        new_file_path = os.path.join(TEMP_VIDEO_DIR, f'stream_{url_hash}.mp4')
        temp_download_path = new_file_path + '.downloading'
        
        print(f"[VideoCache] Downloading: {url}")
        print(f"[VideoCache] Target: {new_file_path}")
        
        # Download with progress
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        with video_cache_lock:
                            video_cache['download_progress'] = progress
                        if progress % 10 == 0:  # Log every 10%
                            print(f"[VideoCache] Download progress: {progress}%")
        
        print(f"[VideoCache] Download complete: {downloaded} bytes")
        
        # Move to final location
        # If file already exists and might be in use, try to remove it carefully
        if os.path.exists(new_file_path):
            try:
                os.remove(new_file_path)
                print(f"[VideoCache] Removed existing file: {new_file_path}")
            except Exception as e:
                # File is in use, can't replace it
                # UPDATE: We'll update the timestamp anyway so we don't keep re-downloading
                print(f"[VideoCache] Cannot replace file (in use): {e}")
                print(f"[VideoCache] Updating cache timestamp to avoid re-downloading")
                
                download_timestamp = time.time()
                
                # Update metadata with NEW timestamp even though we couldn't replace the file
                with video_cache_lock:
                    video_cache['url'] = url
                    video_cache['file_path'] = new_file_path  # Keep using existing file
                    video_cache['download_time'] = download_timestamp  # NEW TIMESTAMP!
                    video_cache['is_downloading'] = False
                    video_cache['download_progress'] = 100
                    video_cache['is_ready'] = True
                
                # Save NEW timestamp to metadata
                update_cache_entry(url, new_file_path, download_timestamp)
                
                # Remove the temp download file since we can't use it
                try:
                    os.remove(temp_download_path)
                    print(f"[VideoCache] Removed temp download: {temp_download_path}")
                except:
                    pass
                
                return new_file_path
        
        os.rename(temp_download_path, new_file_path)
        
        download_timestamp = time.time()
        
        # Update cache and clean old file
        with video_cache_lock:
            old_file = cache_entry.get('file_path') if cache_entry else None
            
            # Delete old cached file if it's different
            if old_file and old_file != new_file_path and os.path.exists(old_file):
                try:
                    print(f"[VideoCache] Removing old cache: {old_file}")
                    os.remove(old_file)
                except Exception as e:
                    print(f"[VideoCache] Failed to remove old cache (in use): {e}")
            
            video_cache['url'] = url
            video_cache['file_path'] = new_file_path
            video_cache['download_time'] = download_timestamp
            video_cache['is_downloading'] = False
            video_cache['download_progress'] = 100
            video_cache['is_ready'] = True
        
        # Save to metadata file
        update_cache_entry(url, new_file_path, download_timestamp)
        
        # Cleanup orphaned files
        cleanup_old_caches()
        
        return new_file_path
        
    except Exception as e:
        print(f"[VideoCache] Download failed: {e}")
        with video_cache_lock:
            video_cache['is_downloading'] = False
            video_cache['error'] = str(e)
            # Don't set is_ready to True on error - let frontend handle it
        
        # Clean up partial download
        if os.path.exists(temp_download_path):
            try:
                os.remove(temp_download_path)
            except:
                pass
        
        return None


def start_cache_refresh_thread(url):
    """Start background thread to refresh cache every 60 minutes."""
    def refresh_worker():
        while True:
            time.sleep(CACHE_REFRESH_INTERVAL)
            
            # Check if URL is still active
            with video_cache_lock:
                if video_cache['url'] != url:
                    print(f"[VideoCache] Refresh thread stopping (URL changed)")
                    break
            
            print(f"[VideoCache] Auto-refresh triggered (60 min elapsed)")
            new_path = download_and_cache_stream(url, force_refresh=True)
            
            if new_path:
                print(f"[VideoCache] Cache refreshed successfully")
                # Reinitialize video capture with new file
                if new_path != current_video_source:
                    initialize_video_capture(new_path)
    
    thread = threading.Thread(target=refresh_worker, daemon=True)
    thread.start()
    print(f"[VideoCache] Started auto-refresh thread (60 min interval)")


# ---------------------------
# Globals for mask accumulation and feed control
# ---------------------------
final_mask = None
road_polygon = None
mask_accumulated = False
feed_running = True  # Start video feed automatically for React frontend
auto_mode = False
calibration_in_progress = False
tracked_vehicles = []  # Store tracked vehicle bounding boxes

# Speed overlay globals
speed_mode = False
speed_frame = None
speed_lock = threading.Lock()
track_state = {}  # track_id -> per-ID state for speeds and crossings
video_fps = 0.0
video_native_fps = 0.0  # Store original video FPS for accurate speed calculations
measured_fps = 0.0  # Runtime processing FPS estimate
frame_idx = 0
current_speed_limit = 75  # Store loaded speed limit (default: 75 km/h)

# Speed streaming buffer (for smooth delayed playback)
SPEED_FEED_DELAY_S = 1.0  # 1 second delayed stream for smoother viewing
speed_buffer = deque(maxlen=300)  # holds tuples (ts, jpg_bytes)

# Use only article methodology for speed (no homography-based per-frame speeds)
USE_HOMOGRAPHY_FOR_SPEED = False
SPEED_AVG_WINDOW_S = 1.0  # compute ~1s rolling average (used for transformed Y history window)

# Detection/tracking config (Roboflow v0.26.1 settings)
CONFIDENCE_THRESHOLD = 0.3  # Roboflow uses 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 1280  # High quality detection (reduce to 960 or 640 if GPU memory issues persist)
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8l.onnx")  # ONNX model for CPU optimization

# When speed feed is running, optionally pause base preview feeds to save CPU
BASE_PREVIEW_WHEN_SPEED = False  # stop original/model feeds while YOLO speed feed is live

# Speed transform (Roboflow-style): map band [A,B] to vertical meters
class SpeedViewTransformer:
    def __init__(self, src_pts: np.ndarray, distance_m: float, target_width_m: float = 25.0):
        # src_pts: 4x2 float32 in image: [ [xA0,yA], [xA1,yA], [xB1,yB], [xB0,yB] ]
        self.distance_m = float(distance_m)
        self.target_width_m = float(target_width_m)
        source = src_pts.astype(np.float32)
        target = np.array([
            [0, 0],
            [self.target_width_m - 1.0, 0],
            [self.target_width_m - 1.0, self.distance_m - 1.0],
            [0, self.distance_m - 1.0]
        ], dtype=np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)

speed_transformer = None
_speed_transformer_meta = None  # (la, lb, cropped_polygon_hash, distance_m)

def ensure_speed_transformer(img_width: int, img_height: int = None, cropped_polygon: np.ndarray = None):
    """
    Create speed transformer using CROPPED POLYGON (not full width).
    This matches Roboflow approach where SOURCE region is the actual road area.
    
    Args:
        img_width: Image width (for fallback only)
        cropped_polygon: 4-point polygon between lines A & B (actual speed zone)
    """
    global speed_transformer, _speed_transformer_meta
    la = calib.get('line_A_y')
    lb = calib.get('line_B_y')
    dist_m = float(calib.get('calib_distance_m', 0.0) or 0.0)
    if not la or not lb or dist_m <= 0:
        speed_transformer = None
        _speed_transformer_meta = None
        return None
    
    # Use cropped polygon if available (CORRECT approach)
    if cropped_polygon is not None and len(cropped_polygon) == 4:
        # Create unique hash for this polygon configuration
        poly_hash = hash(tuple(cropped_polygon.flatten()))
        meta = (int(la), int(lb), poly_hash, float(dist_m))
        
        if _speed_transformer_meta == meta and speed_transformer is not None:
            return speed_transformer
        
        # Use actual cropped polygon as source (Roboflow method)
        # Ensure points are in order: top-left, top-right, bottom-right, bottom-left
        src = cropped_polygon.astype(np.float32)
        # Clamp polygon to image bounds if height provided to avoid out-of-bounds
        if img_height is not None and img_width is not None:
            h, w = float(img_height), float(img_width)
            src[:, 0] = np.clip(src[:, 0], 0.0, w - 1.0)
            src[:, 1] = np.clip(src[:, 1], 0.0, h - 1.0)
        
        # Use road width from Gemini calibration (more accurate than estimation)
        target_width_m = float(calib.get('road_width_m', 10.0))
        
        try:
            speed_transformer = SpeedViewTransformer(src, distance_m=dist_m, target_width_m=target_width_m)
            _speed_transformer_meta = meta
            print(f"[SpeedTransform] ✅ Using CROPPED trapezoid: {dist_m:.1f}m distance, {target_width_m:.1f}m width (from Gemini)")
            return speed_transformer
        except Exception as e:
            print(f"[SpeedTransform] ❌ Error with cropped polygon: {e}")
            speed_transformer = None
            _speed_transformer_meta = None
            return None
    
    # Fallback: full width band (OLD INCORRECT METHOD)
    else:
        meta = (int(la), int(lb), 0, float(dist_m))
        if _speed_transformer_meta == meta and speed_transformer is not None:
            return speed_transformer
        yA = float(la)
        yB = float(lb)
        # Source quad: full width band between A and B
        src = np.array([
            [0.0, yA],
            [float(img_width - 1), yA],
            [float(img_width - 1), yB],
            [0.0, yB]
        ], dtype=np.float32)
        try:
            speed_transformer = SpeedViewTransformer(src, distance_m=dist_m, target_width_m=25.0)
            _speed_transformer_meta = meta
            print(f"[SpeedTransform] ⚠️ Using FULL WIDTH (fallback): {dist_m:.1f}m distance")
        except Exception:
            speed_transformer = None
            _speed_transformer_meta = None
        return speed_transformer


def create_cropped_speed_polygon(road_polygon: np.ndarray, line_a_y: float, line_b_y: float) -> np.ndarray:
    """
    Create a cropped TRAPEZOID polygon for speed detection zone.
    Interpolates left/right boundaries at lines A & B to preserve perspective.
    
    Args:
        road_polygon: Original road polygon from segmentation (Nx2 array)
        line_a_y: Y coordinate of line A (top)
        line_b_y: Y coordinate of line B (bottom)
    
    Returns:
        Cropped trapezoid as 4-point array: [top-left, top-right, bottom-right, bottom-left]
    """
    # Ensure lines are in correct order (top to bottom)
    y_top = min(float(line_a_y), float(line_b_y))
    y_bottom = max(float(line_a_y), float(line_b_y))
    
    # Find left and right boundaries at each Y position by interpolating the polygon edges
    # Strategy: For each Y, find the leftmost and rightmost X coordinates
    
    def find_x_at_y(polygon: np.ndarray, target_y: float) -> tuple:
        """Find leftmost and rightmost X coordinates at a given Y position."""
        x_left = None
        x_right = None
        
        # Iterate through polygon edges
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]  # Next point (wrap around)
            
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            
            # Check if this edge crosses the target Y
            if (y1 <= target_y <= y2) or (y2 <= target_y <= y1):
                # Interpolate X at target_y
                if abs(y2 - y1) > 0.1:  # Avoid division by zero
                    t = (target_y - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    
                    if x_left is None or x_intersect < x_left:
                        x_left = x_intersect
                    if x_right is None or x_intersect > x_right:
                        x_right = x_intersect
        
        # Fallback: if interpolation fails, use min/max from entire polygon
        if x_left is None:
            x_left = float(np.min(polygon[:, 0]))
        if x_right is None:
            x_right = float(np.max(polygon[:, 0]))
        
        return x_left, x_right
    
    # Interpolate boundaries at lines A and B
    x_left_a, x_right_a = find_x_at_y(road_polygon, y_top)
    x_left_b, x_right_b = find_x_at_y(road_polygon, y_bottom)
    
    # Build trapezoid: top-left, top-right, bottom-right, bottom-left
    cropped = np.array([
        [x_left_a, y_top],      # Top-left (narrower)
        [x_right_a, y_top],     # Top-right
        [x_right_b, y_bottom],  # Bottom-right (wider due to perspective)
        [x_left_b, y_bottom]    # Bottom-left
    ], dtype=np.float32)
    return cropped


# ---------------------------
# Frame processing
# ---------------------------

def initialize_video_capture(source):
    """Initialize video capture from file path or HTTP stream URL with smart caching."""
    global cap, video_fps, video_native_fps, current_video_source
    
    if not source:
        return False
    
    if cap is not None:
        cap.release()
    
    print(f"[VideoCapture] Initializing video source: {source}")
    
    # Detect if source is HTTP/HTTPS stream (not a local file)
    is_http_stream = source.lower().startswith('http://') or source.lower().startswith('https://')
    
    if is_http_stream:
        # Use smart caching system
        cached_file = download_and_cache_stream(source)
        
        if cached_file is None:
            print(f"[VideoCapture] ERROR: Failed to download/cache stream")
            with video_cache_lock:
                video_cache['is_ready'] = True  # Mark as ready even on failure
            return False
        
        # Start auto-refresh thread for this URL (only once)
        with video_cache_lock:
            if video_cache['url'] == source:  # Only if this is a new URL
                start_cache_refresh_thread(source)
        
        # Now open the cached file
        source = cached_file
        print(f"[VideoCapture] Using cached file: {source}")
    
    # Open video file (either local or cached)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[VideoCapture] ERROR: Failed to open video source: {source}")
        with video_cache_lock:
            video_cache['is_ready'] = True
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        video_fps = float(fps) if fps and fps > 0 else 25.0
    except Exception:
        video_fps = 25.0
    
    video_native_fps = video_fps
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VideoCapture] Video opened: {width}x{height} @ {video_fps} fps ({total_frames} frames)")
    
    # Wrap with looping capability
    class LoopingVideoCapture:
        def __init__(self, cv_cap):
            self.cv_cap = cv_cap
            self.total_frames = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
        
        def read(self):
            ret, frame = self.cv_cap.read()
            
            # If read failed, try looping back to start
            if not ret:
                print(f"[LoopingCapture] End of video reached at frame {self.current_frame}, looping back...")
                self.cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.cv_cap.read()
            
            # Track current position
            if ret:
                self.current_frame += 1
                # Pre-emptive loop if we're at the last frame
                if self.current_frame >= self.total_frames:
                    print(f"[LoopingCapture] Reached total frames ({self.total_frames}), will loop on next read")
                    self.current_frame = self.total_frames - 1  # Set to last frame marker
            
            return ret, frame
        
        def get(self, prop):
            return self.cv_cap.get(prop)
        
        def release(self):
            self.cv_cap.release()
        
        def isOpened(self):
            return self.cv_cap.isOpened()
    
    cap = LoopingVideoCapture(cap)
    print(f"[VideoCapture] Looping enabled")
    
    with video_cache_lock:
        video_cache['is_ready'] = True
    
    return True


def read_ffmpeg_frame(process, width, height):
    """Read a single frame from ffmpeg stdout pipe."""
    try:
        # Read raw BGR24 frame
        frame_size = width * height * 3
        raw_frame = process.stdout.read(frame_size)
        
        if len(raw_frame) != frame_size:
            return False, None
        
        # Convert raw bytes to numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        return True, frame
    except Exception as e:
        print(f"[VideoCapture] Error reading ffmpeg frame: {e}")
        return False, None


def frame_processor():
    global cap, current_frame, model_frame, current_video_source
    global final_mask, road_polygon, mask_accumulated, feed_running, auto_mode
    global speed_mode, speed_frame, track_state, video_fps, frame_idx
    global video_native_fps, measured_fps

    # Initialize with current source (wait for user to set it via API)
    if current_video_source and initialize_video_capture(current_video_source):
        print(f"[VideoCapture] Successfully initialized with: {current_video_source}")
    else:
        print("[VideoCapture] Waiting for video source to be set via API...")
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 25
    frames_to_accumulate = int(fps) if fps > 0 else 30

    # Runtime FPS estimator
    fps_times = deque(maxlen=120)
    measured_fps = video_fps

    mask_sum = None
    frame_count = 0
    
    last_source_check = current_video_source

    while True:
        if not feed_running:
            time.sleep(0.1)
            continue
        
        # Check if video source changed
        if current_video_source != last_source_check:
            print(f"[VideoCapture] Source changed from {last_source_check} to {current_video_source}")
            if initialize_video_capture(current_video_source):
                last_source_check = current_video_source
                # Reset accumulation state
                mask_sum = None
                frame_count = 0
                mask_accumulated = False
                frames_to_accumulate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        # Support for our FFmpegCapture wrapper
        if hasattr(cap, 'is_ffmpeg') and getattr(cap, 'is_ffmpeg'):
            ret, frame = cap.read()
        else:
            ret, frame = cap.read() if cap else (False, None)
        
        if not ret:
            # If no video source is set yet, wait silently
            if not current_video_source:
                time.sleep(0.5)
                continue
            
            # Check if this is a LoopingVideoCapture (should handle loops internally)
            if isinstance(cap, type) and hasattr(cap, 'cv_cap'):
                # LoopingVideoCapture should handle looping, if we still get False it's a real error
                print("[VideoCapture] LoopingVideoCapture failed to read, attempting to reopen...")
            else:
                print("[VideoCapture] Failed to read frame, attempting to reopen...")
            
            # Try to reopen the source (works for files, may fail for dead streams)
            if initialize_video_capture(current_video_source):
                continue
            else:
                time.sleep(1)
                continue
        # Increment global frame index for timing
        frame_idx += 1

        # Update FPS estimate over ~1s window (for display/streaming only, not speed calc)
        now_ts = time.time()
        fps_times.append(now_ts)
        if len(fps_times) >= 2:
            dt_span = fps_times[-1] - fps_times[0]
            if dt_span >= 1.0:
                measured = (len(fps_times) - 1) / dt_span
                # Smooth update to avoid jitter
                measured_fps = 0.7 * measured_fps + 0.3 * measured

        # Always update current_frame (lightweight copy) for downstream consumers
        with frame_lock:
            current_frame = frame.copy()

        # Avoid extra work for segmentation/model preview while speed mode is active
        if auto_mode and not speed_mode:
            # Model/segmentation mode
            if not mask_accumulated:
                # Step 1: accumulate masks for ~1 second
                road_mask = track.get_road_mask(frame, seg_model)
                if mask_sum is None:
                    mask_sum = road_mask.astype(np.uint16)
                else:
                    mask_sum += road_mask.astype(np.uint16)
                frame_count += 1
                if frame_count >= frames_to_accumulate:
                    avg_mask = (mask_sum / frame_count).astype(np.uint8)
                    _, final_mask = cv2.threshold(avg_mask, 127, 255, cv2.THRESH_BINARY)
                    road_polygon = track.get_road_polygon(final_mask)
                    mask_accumulated = True
                    print("✅ Final road mask & polygon computed.")
                    # --- Take screenshot and call Gemini immediately after polygon computation ---
                    # Save screenshot for line placement under calib folder
                    screenshot_path = os.path.join(CALIB_DIR, 'model_frame_ss.jpg')
                    # Use the current frame for screenshot
                    blended = track.apply_mask(frame, final_mask)
                    if road_polygon is not None:
                        blended = track.draw_polygon(blended, road_polygon)
                    cv2.imwrite(screenshot_path, blended)
                    # Call Gemini with road polygon bounds for accurate line placement
                    gemini_api_key = os.environ.get('GEMINI_API_KEY')
                    if not gemini_api_key:
                        print("[AutoCalib] WARNING: GEMINI_API_KEY not set - using fallback")
                        gemini_result = {"line_A_y": None, "line_B_y": None, "road_width_m": None}
                    else:
                        gemini_result = gemini_lines.get_gemini_line_estimates(
                            screenshot_path, 
                            gemini_api_key,
                        road_polygon=road_polygon  # Pass polygon bounds
                    )
                    print(f"🤖 Gemini result (frame_processor): {gemini_result}")
                    # Update calib lines if Gemini returns valid values
                    if gemini_result.get('line_A_y') is not None and gemini_result.get('line_B_y') is not None:
                        calib['line_A_y'] = int(gemini_result['line_A_y'])
                        calib['line_B_y'] = int(gemini_result['line_B_y'])
                        print(f"✅ Updated calib lines: A={calib['line_A_y']}, B={calib['line_B_y']}")
                    # Update road width if Gemini provides it
                    if gemini_result.get('road_width_m') is not None:
                        calib['road_width_m'] = float(gemini_result['road_width_m'])
                        print(f"✅ Updated road width: {calib['road_width_m']:.1f}m")
                blended = track.apply_mask(frame, road_mask)
                # Draw lines if set
                if calib['line_A_y'] is not None:
                    cv2.line(blended, (0, int(calib['line_A_y'])), (blended.shape[1], int(calib['line_A_y'])), (255,0,0), 2)
                if calib['line_B_y'] is not None:
                    cv2.line(blended, (0, int(calib['line_B_y'])), (blended.shape[1], int(calib['line_B_y'])), (0,0,255), 2)
                with model_lock:                                         
                    model_frame = blended.copy()
            else:
                # Step 2: use final polygon for rest of video
                if final_mask is not None:
                    blended = track.apply_mask(frame, final_mask)
                else:
                    blended = frame.copy()
                if road_polygon is not None:
                    blended = track.draw_polygon(blended, road_polygon)
                # Draw lines if set
                if calib['line_A_y'] is not None:
                    cv2.line(blended, (0, int(calib['line_A_y'])), (blended.shape[1], int(calib['line_A_y'])), (255,0,0), 2)
                if calib['line_B_y'] is not None:
                    cv2.line(blended, (0, int(calib['line_B_y'])), (blended.shape[1], int(calib['line_B_y'])), (0,0,255), 2)
                with model_lock:
                    model_frame = blended.copy()
        else:
            # If not in auto mode, push original frame only if base preview allowed
            if BASE_PREVIEW_WHEN_SPEED or not speed_mode:
                with model_lock:
                    model_frame = frame.copy()

        # Speed overlay processing (uses current_frame to avoid double reading cap)
        # ROBOFLOW-ALIGNED IMPLEMENTATION:
        # 1. High resolution inference (1280px) for stable detections
        # 2. Higher confidence threshold (0.45) to filter weak detections
        # 3. ByteTrack with proper track_thresh, track_buffer, match_thresh
        # 4. Always apply NMS to remove duplicate boxes
        # 5. Outlier rejection (max 5m movement per frame)
        # 6. Minimum 2 seconds tracking before showing speed
        # 7. Sanity checks on distance and speed values
        if speed_mode:
            try:
                from ultralytics import YOLO
                
                # Lazy-load ONNX model once (CPU optimized)
                if not hasattr(frame_processor, "yolo_model"):
                    print(f"[Speed] 📦 Loading ONNX model for CPU inference: {MODEL_PATH}")
                    frame_processor.yolo_model = YOLO(MODEL_PATH, task='detect')
                    print("[Speed] ✅ ONNX model loaded - CPU optimized for fast inference")
                    # Initialize frame skip counter for CPU optimization
                    frame_processor.detection_frame_counter = 0
                
                yolo = frame_processor.yolo_model
                
                # CPU OPTIMIZATION: Skip frames for faster processing
                # Process detection every 2 frames (15 FPS detection on 30 FPS video)
                # Tracking will interpolate between detections
                frame_processor.detection_frame_counter = getattr(frame_processor, 'detection_frame_counter', 0) + 1
                should_detect = (frame_processor.detection_frame_counter % 2 == 0)

                with frame_lock:
                    src = current_frame.copy() if current_frame is not None else None
                if src is None:
                    time.sleep(0.01)
                    continue

                # Draw calibration lines and distance label
                overlay = src.copy()
                if calib['line_A_y'] is not None:
                    cv2.line(overlay, (0, int(calib['line_A_y'])), (overlay.shape[1], int(calib['line_A_y'])), (255,0,0), 2)
                if calib['line_B_y'] is not None:
                    cv2.line(overlay, (0, int(calib['line_B_y'])), (overlay.shape[1], int(calib['line_B_y'])), (0,0,255), 2)
                # Title/status bar background
                header_text = f"Calib distance: {float(calib['calib_distance_m']):.2f} m  |  Processing: {measured_fps:.1f} fps  |  Video: {video_native_fps:.1f} fps"
                (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(overlay, (8, 8), (8+tw+12, 8+th+12), (0,0,0), thickness=-1)
                cv2.putText(overlay, header_text, (14, 8+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # YOLO detection + ByteTrack tracking (Roboflow v0.26.1 approach)
                # Initialize ByteTrack with simple parameters (Roboflow style)
                if not hasattr(frame_processor, "byte_tracker"):
                    # Simple initialization matching Roboflow v0.26.1
                    frame_processor.byte_tracker = sv.ByteTrack(
                        frame_rate=max(1, int(video_native_fps))
                    )
                    # Store last detections for frame skipping
                    frame_processor.last_detections = sv.Detections.empty()
                
                byte_track = frame_processor.byte_tracker

                # Run detection with ONNX (CPU optimized) - ONLY on skip frames
                if should_detect:
                    try:
                        det_result = yolo(src, imgsz=MODEL_RESOLUTION, verbose=False)[0]
                        detections = sv.Detections.from_ultralytics(det_result)
                        
                        # Filter by confidence and exclude person class (class_id=0) - Roboflow approach
                        if hasattr(detections, 'confidence'):
                            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
                        if hasattr(detections, 'class_id'):
                            # Exclude person class (0), keep vehicles only
                            detections = detections[detections.class_id != 0]
                        
                        # Store for interpolation on skipped frames
                        frame_processor.last_detections = detections
                    except Exception as e:
                        print(f"[Speed] ❌ Detection failed: {e}")
                        continue
                else:
                    # Use last detections on skipped frames (ByteTrack will handle interpolation)
                    detections = frame_processor.last_detections
                
                # Filter detections by polygon zone (Roboflow v0.26.1 approach)
                la = calib.get('line_A_y')
                lb = calib.get('line_B_y')
                now = time.time()
                now_frame = frame_idx
                
                # Store cropped polygon for speed transformation
                cropped_polygon_for_transform = None
                
                # Create cropped polygon and filter detections (Roboflow approach)
                if road_polygon is not None and la is not None and lb is not None and len(detections) > 0:
                    # Build cropped polygon dynamically each time (in case lines change)
                    cropped_polygon = create_cropped_speed_polygon(road_polygon, la, lb)
                    # Clamp polygon to frame bounds to avoid any out-of-bounds issues
                    h_, w_ = overlay.shape[0], overlay.shape[1]
                    cropped_polygon = cropped_polygon.astype(np.float32)
                    cropped_polygon[:, 0] = np.clip(cropped_polygon[:, 0], 0.0, w_ - 1.0)
                    cropped_polygon[:, 1] = np.clip(cropped_polygon[:, 1], 0.0, h_ - 1.0)
                    # Save float polygon for transformer
                    cropped_polygon_for_transform = cropped_polygon.copy()
                    # Integer polygon for zone operations
                    cropped_polygon_int = np.clip(np.round(cropped_polygon), [0, 0], [w_ - 1, h_ - 1]).astype(np.int32)
                    
                    # Create polygon zone - Roboflow v0.26.1 simple API
                    if not hasattr(frame_processor, "polygon_zone") or frame_processor.polygon_zone is None or \
                       not hasattr(frame_processor, "last_polygon_bounds") or \
                       frame_processor.last_polygon_bounds != (la, lb):
                        # Simple PolygonZone initialization (v0.26.1)
                        frame_processor.polygon_zone = sv.PolygonZone(polygon=cropped_polygon_int)
                        frame_processor.last_polygon_bounds = (la, lb)
                    polygon_zone = frame_processor.polygon_zone
                    
                    # Filter detections inside polygon (Roboflow v0.26.1 API)
                    mask = polygon_zone.trigger(detections)
                    detections = detections[mask]
                elif la and lb and len(detections) > 0:
                    # Fallback: band filtering - ALLOW BIDIRECTIONAL TRAFFIC
                    # Expand beyond line B to detect opposite-direction vehicles
                    miny = min(float(la), float(lb))
                    maxy = max(float(la), float(lb))
                    band_height = maxy - miny
                    
                    # Extend detection zone beyond line B by same distance (for opposite direction)
                    extended_max = maxy + band_height
                    
                    anchors_check = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    ys = anchors_check[:, 1] if anchors_check is not None and len(anchors_check) == len(detections) else None
                    if ys is not None:
                        tol = 3.0
                        # Allow vehicles from line A to extended zone beyond line B
                        band_mask = (ys >= (miny - tol)) & (ys <= (extended_max + tol))
                        detections = detections[band_mask]
                        print(f"[Speed] Bidirectional detection: {miny:.0f} to {extended_max:.0f} (line A-B: {miny:.0f}-{maxy:.0f})")
                
                # Apply NMS AFTER polygon filtering (Roboflow order)
                if len(detections) > 0:
                    detections = detections.with_nms(threshold=IOU_THRESHOLD)

                # Update tracker
                detections = byte_track.update_with_detections(detections)
                
                # Debug: Log detection count periodically
                if not hasattr(frame_processor, "_detection_log_counter"):
                    frame_processor._detection_log_counter = 0
                frame_processor._detection_log_counter += 1
                if frame_processor._detection_log_counter % 60 == 0:  # Every 60 frames (~2.5 seconds)
                    print(f"[Speed] Tracking {len(detections)} vehicles (filtered: {len(detections)} after polygon/NMS)")
                    if len(detections) > 0 and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                        active_ids = [int(tid) for tid in detections.tracker_id if tid is not None]
                        print(f"  Active IDs: {active_ids}")
                        if hasattr(frame_processor, "coordinates"):
                            for tid in active_ids:
                                coord_count = len(frame_processor.coordinates.get(tid, []))
                                print(f"    ID {tid}: {coord_count} coordinates stored")


                # Ensure we have a speed transformer mapping band [A,B] to meters
                # CRITICAL: Pass cropped polygon for accurate transformation
                transformer_ready = ensure_speed_transformer(overlay.shape[1], overlay.shape[0], cropped_polygon=cropped_polygon_for_transform)
                if transformer_ready is None:
                    # Log warning once per session
                    if not hasattr(frame_processor, "_logged_no_transformer"):
                        print("[Speed] ⚠️ Speed transformer not ready. Ensure calibration lines and distance are set.")
                        print(f"  Line A: {calib.get('line_A_y')}, Line B: {calib.get('line_B_y')}, Distance: {calib.get('calib_distance_m')}m")
                        frame_processor._logged_no_transformer = True

                # Initialize annotators with Roboflow v0.26.1 settings
                if not hasattr(frame_processor, "trace_annotator"):
                    # Roboflow exact settings
                    thickness = 2
                    text_scale = 1  # Roboflow uses text_scale=1
                    
                    frame_processor.box_annotator = sv.BoxAnnotator(thickness=thickness)
                    frame_processor.label_annotator = sv.LabelAnnotator(
                        text_scale=text_scale,
                        text_thickness=thickness,
                        text_position=sv.Position.BOTTOM_CENTER
                    )
                    frame_processor.trace_annotator = sv.TraceAnnotator(
                        thickness=thickness,
                        trace_length=int(video_native_fps * 2),  # 2 second trace
                        position=sv.Position.BOTTOM_CENTER
                    )
                
                trace_annotator = frame_processor.trace_annotator
                box_annotator = frame_processor.box_annotator
                label_annotator = frame_processor.label_annotator

                if len(detections) > 0:
                    # Roboflow approach: track Y coordinates in transformed space
                    anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    
                    # Transform anchors to target space where Y is in meters
                    pts = anchors if anchors is not None else None
                    pts_m = None
                    if pts is not None and len(pts) > 0 and speed_transformer is not None:
                        try:
                            pts_m = speed_transformer.transform_points(points=pts)
                        except Exception as e:
                            print(f"[Speed] Transform error: {e}")
                            pts_m = None
                    
                    # Initialize coordinate tracking dict (Roboflow: exactly 1 second window)
                    if not hasattr(frame_processor, "coordinates"):
                        frame_processor.coordinates = defaultdict(lambda: deque(maxlen=int(video_native_fps)))
                    coordinates = frame_processor.coordinates
                    
                    # Initialize speed history for smoothing (store last 5 speed readings)
                    if not hasattr(frame_processor, "speed_history"):
                        frame_processor.speed_history = defaultdict(lambda: deque(maxlen=5))
                    speed_history = frame_processor.speed_history
                    
                    # Store Y coordinates for tracked vehicles (simple approach)
                    if pts_m is not None and len(pts_m) > 0:
                        for tracker_id, point in zip(detections.tracker_id, pts_m):
                            if tracker_id is not None and point is not None and len(point) >= 2:
                                tid = int(tracker_id)
                                coordinates[tid].append(float(point[1]))  # Store Y coordinate in meters
                    
                    # Format labels (Roboflow v0.26.1 exact formula with smoothing)
                    labels = []
                    for tracker_id in detections.tracker_id:
                        tid = int(tracker_id)
                        
                        # Simplified: Just require minimum tracking time (0.5 seconds)
                        if len(coordinates[tid]) < video_native_fps / 2:
                            labels.append(f"#{tid}")
                        else:
                            # Roboflow formula: first-to-last distance / time
                            y_start = coordinates[tid][-1]  # Most recent
                            y_end = coordinates[tid][0]     # Oldest
                            distance = abs(y_start - y_end)
                            time_seconds = len(coordinates[tid]) / video_native_fps
                            
                            # Only show speed if vehicle has moved (avoid division issues)
                            if distance > 0.1:  # At least 10cm movement
                                raw_speed = distance / time_seconds * 3.6  # m/s to km/h
                                
                                # Sanity check: reject unrealistic speeds (>200 km/h or < 5 km/h)
                                if raw_speed > 200 or raw_speed < 5:
                                    labels.append(f"#{tid}")
                                    continue
                                
                                # Apply exponential moving average for smoothing
                                speed_history[tid].append(raw_speed)
                                
                                # Outlier detection: if speed changes by >30% from previous, it's likely noise
                                if len(speed_history[tid]) >= 2:
                                    prev_speed = speed_history[tid][-2]
                                    speed_change_pct = abs(raw_speed - prev_speed) / prev_speed * 100
                                    if speed_change_pct > 30:  # More than 30% change
                                        # Use previous speed instead of this outlier
                                        speed_history[tid][-1] = prev_speed * 0.9 + raw_speed * 0.1  # Smooth transition
                                
                                # Use weighted average (more weight to recent readings)
                                # This prevents wild fluctuations like 56 -> 62 -> 58
                                if len(speed_history[tid]) >= 3:
                                    # Weighted average: 50% current, 30% previous, 20% older
                                    weights = [0.5, 0.3, 0.2]
                                    recent_speeds = list(speed_history[tid])[-3:]
                                    smoothed_speed = sum(s * w for s, w in zip(recent_speeds, weights))
                                elif len(speed_history[tid]) == 2:
                                    # Simple average of last 2
                                    smoothed_speed = sum(speed_history[tid]) / 2
                                else:
                                    # Not enough history, use raw value
                                    smoothed_speed = raw_speed
                                
                                labels.append(f"#{tid} {int(smoothed_speed)} km/h")
                                
                                # 🔹 NEW: Check for speed violations
                                with violation_recorder_lock:
                                    if violation_recorder is not None:
                                        # Get vehicle class from detection
                                        det_index = list(detections.tracker_id).index(tracker_id)
                                        vehicle_class_id = detections.class_id[det_index] if hasattr(detections, 'class_id') else 0
                                        vehicle_class = CLASS_NAMES_DICT.get(vehicle_class_id, "vehicle")
                                        
                                        # Get bounding box for this detection
                                        bbox = detections.xyxy[det_index] if hasattr(detections, 'xyxy') else None
                                        
                                        # Check violation with bbox and frame (OCR will be done internally)
                                        violation_recorder.check_violation(
                                            vehicle_id=tid,
                                            vehicle_class=vehicle_class,
                                            speed_kmh=smoothed_speed,
                                            bbox=bbox,
                                            frame=overlay.copy(),  # Pass current annotated frame
                                            frame_idx=now_frame  # Pass current frame index for timeout calculation
                                        )
                            else:
                                labels.append(f"#{tid}")
                    
                    # Annotate frame (Roboflow visualization style)
                    overlay = trace_annotator.annotate(scene=overlay, detections=detections)
                    overlay = box_annotator.annotate(scene=overlay, detections=detections)
                    overlay = label_annotator.annotate(scene=overlay, detections=detections, labels=labels)

                # 🔹 NEW: Record annotated frame for violation evidence (AFTER all overlays are added)
                with violation_recorder_lock:
                    if violation_recorder is not None:
                        violation_recorder.add_frame(overlay.copy(), timestamp=time.time())

                # Encode once and push to buffer for delayed streaming
                ok, buf = cv2.imencode('.jpg', overlay)
                if ok:
                    jpg_bytes = buf.tobytes()
                    speed_buffer.append((time.time(), jpg_bytes))
                    with speed_lock:
                        speed_frame = overlay
            except Exception as se:
                # Keep speed mode resilient
                print(f"[Speed] Error: {se}")
                with speed_lock:
                    speed_frame = None

        time.sleep(0.03)  # ~30fps


# ---------------------------
# Flask routes
# ---------------------------

@app.route('/')
def index():
    # API info - frontend is hosted separately on Vercel
    return {
        "service": "NTCS Calibration API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/status",
            "cameras": "/api/cameras",
            "calibration": "/api/calibrate"
        },
        "frontend": "Hosted separately on Vercel"
    }, 200


# Original feed endpoint
@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame, speed_mode
        while True:
            # If speed feed is active and base preview is disabled, skip encoding
            if speed_mode and not BASE_PREVIEW_WHEN_SPEED:
                time.sleep(0.15)
                continue
            with frame_lock:
                frame = current_frame.copy() if current_frame is not None else None
            if frame is None:
                blank = np.zeros((384, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank)
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Model feed endpoint
@app.route('/model_feed')
def model_feed():
    def generate():
        global model_frame, speed_mode
        while True:
            # If speed feed is active and base preview is disabled, skip encoding
            if speed_mode and not BASE_PREVIEW_WHEN_SPEED:
                time.sleep(0.15)
                continue
            with model_lock:
                frame = model_frame.copy() if model_frame is not None else None
            if frame is None:
                blank = np.zeros((384, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank)
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Speed feed endpoint (with fallback to original frame when speed not running)
@app.route('/speed_feed')
def speed_feed():
    def generate():
        global speed_frame, speed_buffer, video_native_fps, current_frame, speed_mode, speed_lock
        delay_s = SPEED_FEED_DELAY_S
        send_interval = 1.0 / max(1.0, video_native_fps)
        last_sent = time.time()
        while True:
            now = time.time()
            sent = False
            
            # If speed mode is active, prefer using speed_frame directly (no delay needed)
            if speed_mode:
                with speed_lock:
                    frame = speed_frame.copy() if speed_frame is not None else None
                
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        sent = True
                        last_sent = now
            
            # Fallback: show original frame if speed not running or no speed frame yet
            if not sent:
                with frame_lock:
                    frame = current_frame.copy() if current_frame is not None else None
                
                if frame is None:
                    blank = np.zeros((384, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for video...", (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', blank)
                else:
                    frame_copy = frame.copy()
                    # Add overlay based on speed mode
                    if speed_mode:
                        cv2.putText(frame_copy, "Speed Detection: STARTING...", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    else:
                        cv2.putText(frame_copy, "Speed Detection: STOPPED", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    ret, buffer = cv2.imencode('.jpg', frame_copy)
                
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    sent = True
                    last_sent = now
            
            if not sent:
                time.sleep(0.005)
                continue
            
            # Pace at approx native FPS
            sleep_left = send_interval - (time.time() - last_sent)
            if sleep_left > 0:
                time.sleep(min(0.03, sleep_left))
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_speed', methods=['POST'])
def start_speed():
    global speed_mode, speed_buffer, track_state
    # Reset buffers/state for a clean start
    speed_buffer.clear()
    track_state = {}
    speed_mode = True
    return {"status": "ok", "message": "Speed mode started"}

@app.route('/stop_speed', methods=['POST'])
def stop_speed():
    global speed_mode
    speed_mode = False
    return {"status": "ok", "message": "Speed mode stopped"}

# Feed control endpoints
@app.route('/start_feed', methods=['POST'])
def start_feed():
    global feed_running, mask_accumulated, auto_mode
    feed_running = True
    mask_accumulated = False
    auto_mode = False
    return {"status": "started"}

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global feed_running
    feed_running = False
    return {"status": "stopped"}

@app.route('/set_calib', methods=['POST'])
def set_calib():
    data = None
    try:
        data = request.get_json(force=True)
    except Exception:
        data = request.json
    if not data:
        return {"status": "error", "message": "No data"}, 400
    for k in ['line_A_y', 'line_B_y', 'calib_distance_m']:
        if k in data:
            calib[k] = float(data[k])
    return {"status": "ok", "calib": calib}

@app.route('/get_calib')
def get_calib():
    return calib

@app.route('/auto_calibrate', methods=['POST'])
def auto_calibrate():
    global auto_mode, mask_accumulated, feed_running
    auto_mode = True
    ensure_calib_dir_clean()
    mask_accumulated = False
    feed_running = True  # Ensure feed is running so model preview updates
    # Wait a short time for mask accumulation and polygon computation
    timeout = 3.0
    poll = 0.05
    waited = 0.0
    while not mask_accumulated and waited < timeout:
        time.sleep(poll)
        waited += poll
    # Compose response with actual polygon if available
    polygon_pts = road_polygon.tolist() if road_polygon is not None else None
    screenshot_path = os.path.join(os.path.dirname(__file__), 'model_frame_ss.jpg')
    # Just return the latest calibration and screenshot info
    return {
        "status": "ok",
        "confidence": 0.9 if polygon_pts else 0.0,
        "method": "segmentation+gemini",
        "source_points": polygon_pts,
        "distance_m": calib['calib_distance_m'],
        "line_A_y": calib['line_A_y'],
        "line_B_y": calib['line_B_y'],
        "screenshot_path": screenshot_path,
        "note": "Auto calibration complete. See logs for Gemini result."
    }

@app.route('/calibrate_distance', methods=['POST'])
def calibrate_distance():
    """
    Trigger perspective-aware distance calibration using detected vehicles.
    Waits for a vehicle to pass between the lines and uses its dimensions for calibration.
    """
    global calibration_in_progress, tracked_vehicles
    
    if calibration_in_progress:
        return {"status": "error", "message": "Calibration already in progress"}, 400
    
    if calib['line_A_y'] is None or calib['line_B_y'] is None:
        return {"status": "error", "message": "Calibration lines not set. Run auto_calibrate first."}, 400
    
    calibration_in_progress = True
    print("[DistanceCalib] Waiting for vehicle between lines...")
    
    try:
        # Load YOLO model for vehicle detection
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATH)  # Use ONNX for CPU optimization
        
        # Wait for a vehicle to pass between the lines
        max_wait = 30  # seconds
        start_time = time.time()
        best_vehicle = None
        best_frame_path = None
        
        while time.time() - start_time < max_wait:
            if not feed_running:
                time.sleep(0.1)
                continue
            
            # Get current frame
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                frame = current_frame.copy()
            
            # Detect vehicles
            results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)  # car, motorcycle, bus, truck
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    detections.append(box.tolist())
            
            # Find vehicle between lines
            vehicle_bbox = perspective_calibration.find_vehicle_between_lines(
                frame, calib['line_A_y'], calib['line_B_y'], detections
            )
            
            if vehicle_bbox:
                # Save frame with vehicle
                temp_frame_path = os.path.join(os.path.dirname(__file__), 'vehicle_calib_frame.jpg')
                cv2.imwrite(temp_frame_path, frame)
                best_vehicle = vehicle_bbox
                best_frame_path = temp_frame_path
                print(f"[DistanceCalib] Found suitable vehicle: {vehicle_bbox}")
                break
            
            time.sleep(0.1)
        
        if not best_vehicle:
            calibration_in_progress = False
            return {
                "status": "timeout",
                "message": f"No suitable vehicle found between lines in {max_wait} seconds"
            }
        
        # Perform perspective calibration
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "GEMINI_API_KEY environment variable not set"
            }
        
        # Try to use homography if available (from ITSC paper IPM approach)
        homography = calib.get('homography_matrix')
        if homography is not None:
            # Convert from list back to numpy array
            homography = np.array(homography, dtype=np.float32)
            print("[CalibrateDistance] Using homography-based IPM (most accurate)")
        else:
            print("[CalibrateDistance] No homography available, using vehicle-based method")
        
        calib_result = perspective_calibration.calculate_perspective_calibration(
            best_frame_path,
            calib['line_A_y'],
            calib['line_B_y'],
            best_vehicle,
            gemini_api_key,
            homography_matrix=homography  # 🔹 NEW: Pass homography if available
        )
        
        if calib_result and calib_result.get('distance_meters'):
            # Update calibration
            calib['calib_distance_m'] = calib_result['distance_meters']
            calib['confidence_score'] = calib_result['confidence']
            calib['calibration_method'] = f"perspective_{calib_result['method']}"
            
            print(f"✅ Distance calibrated: {calib['calib_distance_m']:.2f}m (confidence: {calib['confidence_score']:.2f})")
            
            calibration_in_progress = False
            return {
                "status": "ok",
                "distance_m": calib['calib_distance_m'],
                "confidence": calib['confidence_score'],
                "method": calib['calibration_method'],
                "details": calib_result,
                "note": calib_result.get('note', 'Perspective calibration complete')
            }
        else:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Failed to calibrate distance",
                "details": calib_result
            }
    
    except Exception as e:
        calibration_in_progress = False
        print(f"[DistanceCalib] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


@app.route('/calibrate_homography', methods=['POST'])
def calibrate_homography():
    """
    Set homography matrix for IPM-based distance calibration (ITSC paper method).
    
    Requires 4 corresponding points:
    - image_points: 4 points in image coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    - world_points: 4 corresponding points in real-world coordinates (meters) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Example:
    {
        "image_points": [[100, 500], [900, 500], [950, 300], [50, 300]],
        "world_points": [[0, 0], [5, 0], [5, 10], [0, 10]]
    }
    
    Or directly provide the matrix:
    {
        "homography_matrix": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]
    }
    """
    try:
        data = request.get_json()
        
        if 'homography_matrix' in data:
            # Direct matrix provided
            H = np.array(data['homography_matrix'], dtype=np.float32)
            if H.shape != (3, 3):
                return {"status": "error", "message": "Matrix must be 3x3"}, 400
            
            # Store as list for JSON serialization
            calib['homography_matrix'] = H.tolist()
            calib['homography_points'] = None
            print("[HomographyCalib] ✅ Homography matrix set directly")
            print(H)
            
            return {
                "status": "ok",
                "message": "Homography matrix set successfully",
                "matrix": H.tolist()
            }
        
        elif 'image_points' in data and 'world_points' in data:
            # Compute from point correspondences
            img_pts = data['image_points']
            world_pts = data['world_points']
            
            if len(img_pts) != 4 or len(world_pts) != 4:
                return {"status": "error", "message": "Need exactly 4 point pairs"}, 400
            
            H = perspective_calibration.estimate_homography_from_points(
                np.array(img_pts, dtype=np.float32),
                np.array(world_pts, dtype=np.float32)
            )
            
            if H is None:
                return {"status": "error", "message": "Failed to compute homography"}, 500
            
            # Store as list for JSON serialization
            calib['homography_matrix'] = H.tolist()
            calib['homography_points'] = {'image': img_pts, 'world': world_pts}
            print("[HomographyCalib] ✅ Homography computed from 4 point pairs")
            
            return {
                "status": "ok",
                "message": "Homography calibrated successfully",
                "matrix": H.tolist(),
                "points": calib['homography_points']
            }
        
        else:
            return {
                "status": "error",
                "message": "Need either 'homography_matrix' or 'image_points' + 'world_points'"
            }, 400
    
    except Exception as e:
        print(f"[HomographyCalib] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


@app.route('/clear_homography', methods=['POST'])
def clear_homography():
    """Clear homography calibration (fall back to vehicle-based method)"""
    calib['homography_matrix'] = None
    calib['homography_points'] = None
    print("[HomographyCalib] Cleared homography, will use vehicle-based method")
    return {"status": "ok", "message": "Homography cleared"}


@app.route('/get_calibration', methods=['GET'])
def get_calibration():
    """Get current calibration status including homography"""
    return {
        "line_A_y": calib['line_A_y'],
        "line_B_y": calib['line_B_y'],
        "distance_m": calib['calib_distance_m'],
        "confidence": calib['confidence_score'],
        "method": calib['calibration_method'],
        "has_homography": calib['homography_matrix'] is not None,
        "homography_points": calib['homography_points']
    }


@app.route('/auto_calibrate_homography', methods=['POST'])
def auto_calibrate_homography():
    """
    🆕 HYBRID APPROACH: Automatic homography calibration using road polygon + vehicles.
    
    Based on Roboflow method: https://blog.roboflow.com/estimate-speed-computer-vision/
    
    Steps:
    1. Use road polygon from segmentation as SOURCE region
    2. Detect multiple vehicles at different positions
    3. Use vehicle dimensions to estimate road size (TARGET region)
    4. Compute homography transformation matrix
    5. No manual measurement needed!
    """
    global road_polygon, current_frame, calibration_in_progress
    
    try:
        if calibration_in_progress:
            return {"status": "error", "message": "Another calibration is in progress"}, 409
        
        calibration_in_progress = True
        
        print("[AutoHomography] Starting automatic homography calibration...")
        
        # Check if feed is running
        if not feed_running:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Video feed not running. Click 'Start Feed' first."
            }, 400
        
        # Check if we have road polygon
        if road_polygon is None or len(road_polygon) < 4:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Road polygon not available. Steps: 1) Start Feed, 2) Click 'Auto Calibrate (Lines)', 3) Try again."
            }, 400
        
        # Wait for vehicles to appear (collect for 5 seconds)
        print("[AutoHomography] Collecting vehicle data (5 seconds)...")
        
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATH)  # Use ONNX for CPU optimization
        
        collected_vehicles = []
        start_time = time.time()
        collection_duration = 5  # seconds
        
        while time.time() - start_time < collection_duration:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = current_frame.copy()
            
            # Detect vehicles
            results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)  # car, motorcycle, bus, truck
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls.item())
                    class_name = yolo_model.names[cls]
                    
                    # Map YOLO classes to vehicle types
                    vehicle_type_map = {
                        'car': 'sedan',
                        'motorcycle': 'motorcycle',
                        'bus': 'bus',
                        'truck': 'truck'
                    }
                    vehicle_type = vehicle_type_map.get(class_name, 'sedan')
                    
                    bbox = box.xyxy[0].tolist()
                    collected_vehicles.append({
                        'bbox': bbox,
                        'vehicle_type': vehicle_type,
                        'y_position': (bbox[1] + bbox[3]) / 2
                    })
            
            time.sleep(0.2)  # Sample every 200ms
        
        if len(collected_vehicles) < 2:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": f"Not enough vehicles detected ({len(collected_vehicles)}/2 minimum). Need vehicles at different positions."
            }, 400
        
        print(f"[AutoHomography] Collected {len(collected_vehicles)} vehicle samples")
        
        # Remove duplicates (vehicles at similar Y positions)
        unique_vehicles = []
        seen_y_positions = set()
        for v in collected_vehicles:
            y_bucket = int(v['y_position'] / 50)  # Group by 50px buckets
            if y_bucket not in seen_y_positions:
                unique_vehicles.append(v)
                seen_y_positions.add(y_bucket)
        
        print(f"[AutoHomography] Using {len(unique_vehicles)} unique vehicle positions")
        
        if len(unique_vehicles) < 2:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Vehicles at different Y-positions needed for perspective estimation"
            }, 400
        
        # Estimate homography using road polygon + vehicles
        img_height, img_width = frame.shape[:2]
        
        transformer = auto_homography.estimate_homography_from_polygon_and_vehicles(
            road_polygon,
            unique_vehicles,
            perspective_calibration.VEHICLE_DIMENSIONS,
            img_height,
            img_width
        )
        
        if transformer is None:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Failed to compute homography matrix"
            }, 500
        
        # Save homography matrix (convert numpy arrays to lists for JSON serialization)
        calib['homography_matrix'] = transformer.m.tolist()
        calib['homography_points'] = {
            'source': transformer.source.tolist(),
            'target': transformer.target.tolist(),
            'method': 'auto_from_vehicles'
        }
        
        print("[AutoHomography] ✅ Automatic homography calibration successful!")
        
        # Now test distance calculation
        if calib['line_A_y'] and calib['line_B_y']:
            distance = transformer.transform_distance(
                calib['line_A_y'],
                calib['line_B_y'],
                img_width
            )
            
            # Convert numpy types to Python native types for JSON serialization
            distance = float(distance)
            
            print(f"[AutoHomography] Homography distance: {distance:.2f}m")
            
            # 🔹 NEW: Verify distance using LLM-based vehicle size comparison
            try:
                print("[LLM Verification] Attempting vehicle size comparison...")
                
                # Prepare vehicle data for verification (match function signature)
                vehicles_bboxes = []
                vehicle_types = []
                for v in unique_vehicles:
                    bbox = v['bbox']
                    vehicles_bboxes.append(bbox)
                    vehicle_types.append(v['vehicle_type'])
                
                # Get current frame for vehicle cropping
                with frame_lock:
                    verification_frame = current_frame.copy() if current_frame is not None else frame.copy()
                
                # Call LLM verification with homography distance as baseline
                gemini_api_key = os.environ.get('GEMINI_API_KEY')
                if not gemini_api_key:
                    print("[LLM Verification] ⚠️ GEMINI_API_KEY not found, skipping LLM verification")
                    calibrated_distance = distance
                    calibration_method = 'auto_homography_ipm'
                    verification_status = 'skipped_no_api_key'
                else:
                    verification_result = vehicle_size_verification.verify_distance_with_vehicle_comparison(
                        frame=verification_frame,
                        vehicles_bboxes=vehicles_bboxes,
                        vehicle_types=vehicle_types,
                        line_a_y=calib['line_A_y'],
                        line_b_y=calib['line_B_y'],
                        vehicle_dimensions=perspective_calibration.VEHICLE_DIMENSIONS,
                        api_key=gemini_api_key,
                        homography_distance=distance
                    )
                    
                    if verification_result and (
                        'calibrated_distance' in verification_result or 'llm_distance' in verification_result or 'homography_distance' in verification_result
                    ):
                        # Prefer calibrated distance if present, otherwise fallback to llm or homography
                        if 'calibrated_distance' in verification_result:
                            calibrated_distance = verification_result['calibrated_distance']
                            calibration_method = verification_result.get('calibration_method', 'hybrid_homography_llm')
                        elif 'llm_distance' in verification_result and verification_result['llm_distance']:
                            calibrated_distance = verification_result['llm_distance']
                            calibration_method = 'llm_only'
                        else:
                            calibrated_distance = distance
                            calibration_method = 'auto_homography_ipm'

                        verification_status = 'success'

                        print(f"[LLM Verification] ✅ Verification completed")
                        print(f"  - Homography: {distance:.2f}m")
                        print(f"  - LLM estimate: {verification_result.get('llm_distance', 'N/A')}m")
                        print(f"  - Final: {calibrated_distance:.2f}m")
                        print(f"  - Agreement: {verification_result.get('distances_agree', False)}")
                    else:
                        msg = None
                        try:
                            msg = verification_result.get('message') if verification_result else None
                        except Exception:
                            msg = None
                        print(f"[LLM Verification] ⚠️ Verification failed: {msg or 'Unknown error'}")
                        calibrated_distance = distance
                        calibration_method = 'auto_homography_ipm'
                        verification_status = 'failed'
                
            except Exception as e:
                print(f"[LLM Verification] ⚠️ Error during verification: {e}")
                import traceback
                traceback.print_exc()
                calibrated_distance = distance
                calibration_method = 'auto_homography_ipm'
                verification_status = 'error'
            
            # Use calibrated distance (hybrid if LLM succeeded, homography-only if not)
            calib['calib_distance_m'] = calibrated_distance
            calib['calibration_method'] = calibration_method
            calib['confidence_score'] = 0.90  # High confidence with auto homography
            
            calibration_in_progress = False
            return {
                "status": "ok",
                "message": f"Automatic homography calibrated successfully! Calibration method: {calibration_method}",
                "distance_m": calibrated_distance,
                "homography_distance_m": distance,
                "verification_status": verification_status,
                "confidence": 0.90,
                "method": calibration_method,
                "vehicles_used": len(unique_vehicles),
                "matrix": transformer.m.tolist(),
                "source_region": transformer.source.tolist(),
                "target_region_meters": transformer.target.tolist()
            }
        else:
            calibration_in_progress = False
            return {
                "status": "ok",
                "message": "Homography calibrated. Set lines first to measure distance.",
                "matrix": transformer.m.tolist()
            }
    
    except Exception as e:
        calibration_in_progress = False
        print(f"[AutoHomography] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


@app.route('/auto_calibrate_tracked', methods=['POST'])
def auto_calibrate_tracked():
    """
    🆕 TRACKED VEHICLE CALIBRATION (LLM-First Approach)
    
    Superior method that:
    1. Tracks a SINGLE vehicle from line A → line B
    2. Captures screenshots at both positions WITH visible calibration lines
    3. Sends to Gemini Vision with road marking context
    4. LLM estimates real-world distance first
    5. Configures homography to align with LLM's judgment
    
    This is better than random vehicle sampling because:
    - Same vehicle = fair size comparison
    - Lines visible in image = spatial context for LLM
    - Road markings = additional distortion reference
    - LLM-driven = homography follows AI judgment
    """
    global road_polygon, current_frame, calibration_in_progress, calib
    
    try:
        if calibration_in_progress:
            return {"status": "error", "message": "Another calibration is in progress"}, 409
        
        calibration_in_progress = True
        ensure_calib_dir_clean()
        
        print("[TrackedCalibration] Starting tracked vehicle calibration...")
        
        # Check if feed is running
        if not feed_running:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Feed not running. Steps: 1) Start Feed, 2) Try again."
            }, 400
        
        # Check if lines are set
        if not calib['line_A_y'] or not calib['line_B_y']:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Calibration lines not set. Steps: 1) Auto Calibrate (Lines), 2) Try again."
            }, 400
        
        # Check if road polygon exists
        if road_polygon is None or len(road_polygon) < 3:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Road polygon not available. Steps: 1) Start Feed, 2) Wait for segmentation, 3) Try again."
            }, 400
        
        # Get Gemini API key
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "GEMINI_API_KEY not set. Set environment variable and restart server."
            }, 400
        
        line_a_y = calib['line_A_y']
        line_b_y = calib['line_B_y']
        
        print(f"[TrackedCalibration] Waiting for vehicle to cross line A (Y={line_a_y})...")
        
        # Initialize YOLO for vehicle tracking
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATH)  # Use ONNX for CPU optimization
        
        # Track vehicles across frames
        tracked_vehicles = {}  # track_id -> {'frames': [...], 'bboxes': [...], 'type': ...}
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        frame_at_line_a = None
        frame_at_line_b = None
        vehicle_bbox_at_a = None
        vehicle_bbox_at_b = None
        selected_vehicle_type = None
        selected_track_id = None
        
        # Tracking parameters
        line_a_tolerance = 50  # pixels
        line_b_tolerance = 50
        
        while time.time() - start_time < max_wait_time:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = current_frame.copy()
            
            # Run YOLO with tracking
            results = yolo_model.track(
                frame, 
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                persist=True,
                verbose=False
            )
            
            if len(results[0].boxes) == 0:
                time.sleep(0.1)
                continue
            
            # Process detections
            for box in results[0].boxes:
                if box.id is None:
                    continue
                
                track_id = int(box.id.item())
                cls = int(box.cls.item())
                class_name = yolo_model.names[cls]
                
                # Map to vehicle type
                vehicle_type_map = {
                    'car': 'sedan',
                    'motorcycle': 'motorcycle',
                    'bus': 'bus',
                    'truck': 'truck'
                }
                vehicle_type = vehicle_type_map.get(class_name, 'sedan')
                
                bbox = box.xyxy[0].tolist()
                y_center = (bbox[1] + bbox[3]) / 2
                
                # Initialize tracking for this vehicle
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {
                        'type': vehicle_type,
                        'crossed_a': False,
                        'crossed_b': False,
                        'frame_at_a': None,
                        'bbox_at_a': None,
                        'frame_at_b': None,
                        'bbox_at_b': None
                    }
                
                vehicle_data = tracked_vehicles[track_id]
                
                # Check if vehicle crossed line A
                if not vehicle_data['crossed_a']:
                    if abs(y_center - line_a_y) < line_a_tolerance:
                        print(f"[TrackedCalibration] Vehicle {track_id} ({vehicle_type}) crossed line A!")
                        vehicle_data['crossed_a'] = True
                        vehicle_data['frame_at_a'] = frame.copy()
                        vehicle_data['bbox_at_a'] = bbox
                
                # Check if vehicle crossed line B (only if already crossed A)
                elif not vehicle_data['crossed_b']:
                    if abs(y_center - line_b_y) < line_b_tolerance:
                        print(f"[TrackedCalibration] Vehicle {track_id} ({vehicle_type}) crossed line B!")
                        vehicle_data['crossed_b'] = True
                        vehicle_data['frame_at_b'] = frame.copy()
                        vehicle_data['bbox_at_b'] = bbox
                        
                        # Found complete track! Use this vehicle
                        selected_track_id = track_id
                        selected_vehicle_type = vehicle_type
                        frame_at_line_a = vehicle_data['frame_at_a']
                        frame_at_line_b = vehicle_data['frame_at_b']
                        vehicle_bbox_at_a = vehicle_data['bbox_at_a']
                        vehicle_bbox_at_b = vehicle_data['bbox_at_b']
                        
                        print(f"[TrackedCalibration] ✅ Complete track found for vehicle {track_id}!")
                        break
            
            # If we found a complete track, stop waiting
            if selected_track_id is not None:
                break
            
            time.sleep(0.1)
        
        # Check if we found a tracked vehicle
        if selected_track_id is None:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": f"No vehicle completed the journey from line A to line B within {max_wait_time}s. Try again with more traffic."
            }, 400
        
        print(f"[TrackedCalibration] Analyzing tracked {selected_vehicle_type} (ID: {selected_track_id})...")
        
        # Call LLM to estimate distance
        llm_result = tracked_vehicle_calibration.estimate_distance_from_tracked_vehicle(
            frame_at_line_a=frame_at_line_a,
            frame_at_line_b=frame_at_line_b,
            vehicle_bbox_a=vehicle_bbox_at_a,
            vehicle_bbox_b=vehicle_bbox_at_b,
            line_a_y=line_a_y,
            line_b_y=line_b_y,
            vehicle_type=selected_vehicle_type,
            gemini_api_key=gemini_api_key,
            road_polygon=road_polygon
        )
        
        if llm_result['status'] != 'success':
            calibration_in_progress = False
            return {
                "status": "error",
                "message": f"LLM distance estimation failed: {llm_result.get('message', 'Unknown error')}"
            }, 500
        
        llm_distance = llm_result['estimated_distance_meters']
        llm_confidence = llm_result['confidence']
        
        print(f"[TrackedCalibration] LLM estimated distance: {llm_distance:.2f}m (confidence: {llm_confidence:.2f})")
        print(f"  Reasoning: {llm_result.get('reasoning', 'N/A')}")
        
        # Configure homography based on LLM distance
        img_height, img_width = frame_at_line_a.shape[:2]
        
        transformer = tracked_vehicle_calibration.configure_homography_from_llm_distance(
            road_polygon=road_polygon,
            llm_distance_meters=llm_distance,
            line_a_y=line_a_y,
            line_b_y=line_b_y,
            img_width=img_width,
            img_height=img_height,
            road_width_estimate_meters=10.0
        )
        
        if transformer is None:
            calibration_in_progress = False
            return {
                "status": "error",
                "message": "Failed to configure homography from LLM distance"
            }, 500
        
        # Save calibration
        calib['homography_matrix'] = transformer.m.tolist()
        calib['homography_points'] = {
            'source': transformer.source.tolist(),
            'target': transformer.target.tolist(),
            'method': 'tracked_vehicle_llm'
        }
        calib['calib_distance_m'] = llm_distance
        calib['calibration_method'] = 'tracked_vehicle_llm_first'
        calib['confidence_score'] = llm_confidence
        
        print(f"[TrackedCalibration] ✅ Calibration complete!")
        print(f"  Distance: {llm_distance:.2f}m")
        print(f"  Vehicle size ratio: {llm_result['vehicle_size_ratio']:.2f}x")
        print(f"  Pixel distance: {llm_result['pixel_distance']}px")
        
        # Extract detailed analysis
        vehicle_id = llm_result.get('vehicle_identification', {})
        camera_info = llm_result.get('camera_analysis', {})
        distance_calc = llm_result.get('distance_calculation', {})
        
        calibration_in_progress = False
        return {
            "status": "ok",
            "message": f"Tracked vehicle calibration successful! Vehicle traveled {llm_distance:.2f}m between lines.",
            "distance_m": llm_distance,
            "confidence": llm_confidence,
            "method": "tracked_vehicle_llm_first",
            "vehicle_type": selected_vehicle_type,
            "vehicle_size_ratio": llm_result['vehicle_size_ratio'],
            "pixel_distance": llm_result['pixel_distance'],
            "llm_reasoning": llm_result.get('reasoning', ''),
            "warnings": llm_result.get('warnings', 'none'),
            "annotated_frame_a_path": llm_result.get('annotated_frame_a_path'),
            "annotated_frame_b_path": llm_result.get('annotated_frame_b_path'),
            "vehicle_identification": {
                "category": vehicle_id.get('category', selected_vehicle_type),
                "model": vehicle_id.get('specific_model', 'unknown'),
                "length_m": vehicle_id.get('real_length_meters', 0),
                "width_m": vehicle_id.get('real_width_meters', 0),
                "height_m": vehicle_id.get('real_height_meters', 0)
            },
            "camera_analysis": {
                "height_m": camera_info.get('estimated_height_meters', 0),
                "height_category": camera_info.get('height_category', 'unknown'),
                "viewing_angle": camera_info.get('viewing_angle', 'unknown'),
                "fov": camera_info.get('field_of_view', 'unknown'),
                "perspective": camera_info.get('perspective_compression', 'unknown')
            },
            "distance_calculation": {
                "geometric_m": distance_calc.get('method_a_geometric_meters', llm_distance),
                "road_markings_m": distance_calc.get('method_b_road_markings_meters', None),
                "methods_agree": distance_calc.get('methods_agree', True),
                "details": distance_calc.get('calculation_details', '')
            },
            "matrix": transformer.m.tolist(),
            "source_region": transformer.source.tolist(),
            "target_region_meters": transformer.target.tolist()
        }
        
    except Exception as e:
        calibration_in_progress = False
        print(f"[TrackedCalibration] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


# ---------------------------
# Combined Auto Calibration Endpoint (Gemini Lines + Tracked Vehicle)
# ---------------------------
@app.route('/api/auto_calibrate_full', methods=['POST'])
def auto_calibrate_full():
    """
    Combined auto calibration that runs:
    1. Gemini-based line detection
    2. Tracked vehicle calibration for distance
    """
    global calib, calibration_in_progress, auto_mode, mask_accumulated, feed_running, tracked_vehicles, road_polygon
    
    if calibration_in_progress:
        return {"status": "error", "message": "Calibration already in progress"}, 400
    
    calibration_in_progress = True
    
    try:
        print("\n" + "="*60)
        print("🤖 COMBINED AUTO CALIBRATION - Starting...")
        print("="*60)
        
        # Step 1: Trigger Gemini line detection
        print("\n[Step 1/2] Running Gemini line detection...")
        auto_mode = True
        ensure_calib_dir_clean()
        mask_accumulated = False
        feed_running = True
        
        # Wait for mask accumulation and Gemini to process
        print("  Waiting for segmentation and Gemini processing...")
        timeout = 100.0  # Increased timeout for CPU-only segmentation (can take 60-80s) + Gemini (20-30s)
        poll = 0.5
        waited = 0.0
        
        initial_line_a = calib.get('line_A_y')
        initial_line_b = calib.get('line_B_y')
        initial_width = calib.get('road_width_m')
        
        while waited < timeout:
            time.sleep(poll)
            waited += poll
            
            # Check if Gemini updated the lines
            if (calib.get('line_A_y') != initial_line_a or 
                calib.get('line_B_y') != initial_line_b or
                calib.get('road_width_m') != initial_width):
                print(f"  ✅ Gemini updated lines: A={calib['line_A_y']}, B={calib['line_B_y']}")
                print(f"  ✅ Road width: {calib.get('road_width_m', 10.0)}m")
                # Persist the latest road polygon into calib if available
                try:
                    if calib.get('source_points') is None and road_polygon is not None and len(road_polygon) >= 4:
                        calib['source_points'] = np.array(road_polygon, dtype=np.float32).tolist()
                except Exception:
                    pass
                break
        
        if waited >= timeout:
            calibration_in_progress = False
            return {"status": "error", "message": "Gemini line detection timed out after 100 seconds. Check backend logs for details."}, 500
        
        # Step 2: Run tracked vehicle calibration for distance
        print("\n[Step 2/2] Running tracked vehicle calibration...")
        print("  Tracking vehicle from Line A to Line B...")
        
        # Load required modules
        from ultralytics import YOLO
        import tracked_vehicle_calibration
        from supervision import Detections
        from supervision.tracker.byte_tracker.core import ByteTrack
        
        # Initialize tracker and YOLO
        yolo_model = YOLO(MODEL_PATH)  # Use ONNX for CPU optimization
        tracker = ByteTrack(frame_rate=30, lost_track_buffer=90)
        
        # Track vehicles crossing from Line A to Line B
        max_track_time = 60  # 60 seconds to find a complete crossing
        start_time = time.time()
        
        tracked_crossings = {}  # track_id -> {'frame_a': frame, 'bbox_a': bbox, 'frame_b': None, 'bbox_b': None}
        vehicle_types = {}  # track_id -> vehicle_class_name
        
        line_a_y = calib['line_A_y']
        line_b_y = calib['line_B_y']
        
        while time.time() - start_time < max_track_time:
            if not feed_running:
                time.sleep(0.05)
                continue
            
            # Get current frame
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.05)
                    continue
                frame = current_frame.copy()
            
            # Detect vehicles
            results = yolo_model(frame, conf=0.3, classes=[2, 3, 5, 7], verbose=False)  # car, motorcycle, bus, truck
            
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                time.sleep(0.05)
                continue
            
            # Convert to Supervision format
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            
            # Update tracker
            detections = tracker.update_with_detections(detections)
            
            # Check each tracked vehicle
            for i, track_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i]
                vehicle_class = detections.class_id[i]
                x1, y1, x2, y2 = bbox
                center_y = (y1 + y2) / 2
                
                # Map YOLO class to vehicle type
                class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                vehicle_type = class_names.get(vehicle_class, 'car')
                
                # Initialize tracking for new vehicles
                if track_id not in tracked_crossings:
                    tracked_crossings[track_id] = {
                        'frame_a': None,
                        'bbox_a': None,
                        'frame_b': None,
                        'bbox_b': None,
                        'crossed_a': False,
                        'crossed_b': False
                    }
                    vehicle_types[track_id] = vehicle_type
                
                track_data = tracked_crossings[track_id]
                
                # Check if vehicle crosses Line A (tolerance: ±20px)
                if not track_data['crossed_a'] and abs(center_y - line_a_y) < 20:
                    track_data['frame_a'] = frame.copy()
                    track_data['bbox_a'] = bbox.tolist()
                    track_data['crossed_a'] = True
                    print(f"  📸 Vehicle #{track_id} ({vehicle_type}) crossed Line A at y={center_y:.0f}")
                
                # Check if vehicle crosses Line B (tolerance: ±20px)
                elif track_data['crossed_a'] and not track_data['crossed_b'] and abs(center_y - line_b_y) < 20:
                    track_data['frame_b'] = frame.copy()
                    track_data['bbox_b'] = bbox.tolist()
                    track_data['crossed_b'] = True
                    print(f"  📸 Vehicle #{track_id} ({vehicle_type}) crossed Line B at y={center_y:.0f}")
                    
                    # Complete crossing detected!
                    print(f"  ✅ Complete crossing detected for vehicle #{track_id}!")
                    
                    # Send to Gemini for distance estimation
                    gemini_api_key = os.environ.get('GEMINI_API_KEY')
                    if not gemini_api_key:
                        print("[CrossingDetect] WARNING: GEMINI_API_KEY not set, skipping LLM distance validation")
                        source_polygon = calib.get('source_points')
                    if source_polygon is not None:
                        source_polygon = np.array(source_polygon, dtype=np.float32)
                    
                    distance_result = tracked_vehicle_calibration.estimate_distance_from_tracked_vehicle(
                        frame_at_line_a=track_data['frame_a'],
                        frame_at_line_b=track_data['frame_b'],
                        vehicle_bbox_a=track_data['bbox_a'],
                        vehicle_bbox_b=track_data['bbox_b'],
                        line_a_y=line_a_y,
                        line_b_y=line_b_y,
                        vehicle_type=vehicle_types[track_id],
                        gemini_api_key=gemini_api_key,
                        road_polygon=source_polygon
                    )
                    
                    if distance_result['status'] == 'success':
                        distance_m = distance_result['estimated_distance_meters']
                        confidence = distance_result['confidence']
                        
                        calib['calib_distance_m'] = distance_m
                        calib['confidence_score'] = confidence
                        calib['calibration_method'] = 'auto_tracked_vehicle'
                        
                        print(f"  ✅ Distance estimated: {distance_m:.2f}m (confidence: {confidence:.2f})")
                        print(f"  💡 {distance_result.get('reasoning', '')}")
                        
                        # Success! Break out of tracking loop
                        break
                    else:
                        print(f"  ⚠️ Distance estimation failed: {distance_result.get('message', 'Unknown error')}")
                        continue
            
            # Check if we found a successful calibration
            if calib.get('calibration_method') == 'auto_tracked_vehicle':
                break
            
            time.sleep(0.05)
        
        # Fallback if no complete crossing detected
        if calib.get('calibration_method') != 'auto_tracked_vehicle':
            print("  ⚠️ No complete vehicle crossing detected, using default distance")
            calib['calib_distance_m'] = 10.0
            calib['calibration_method'] = 'auto_lines_only'
            calib['confidence_score'] = 0.6
        
        print("\n" + "="*60)
        print("✅ COMBINED AUTO CALIBRATION - Complete!")
        print(f"   Lines: A={calib['line_A_y']}, B={calib['line_B_y']}")
        print(f"   Distance: {calib['calib_distance_m']}m")
        print(f"   Road Width: {calib['road_width_m']}m")
        print("="*60 + "\n")
        
        calibration_in_progress = False
        
        # Save calibration to backend API
        if current_camera_id:
            try:
                stats = {
                    "line_ay": int(calib['line_A_y']),
                    "line_by": int(calib['line_B_y']),
                    "polygon_points": calib.get('source_points', []),
                    "distance": float(calib['calib_distance_m']),
                    "width": float(calib['road_width_m']),
                    "method": calib.get('calibration_method', 'auto_combined'),
                    "confidence": float(calib.get('confidence_score', 0.85)),
                    "reason": "Auto calibration completed"
                }
                
                # Get camera details (you may need to pass these from frontend)
                camera_link = current_video_source
                location = "Unknown"  # Will be updated by frontend
                
                save_calibration_to_backend(current_camera_id, camera_link, location, stats)
            except Exception as save_error:
                print(f"[BackendAPI] Failed to save calibration: {save_error}")
        
        return {
            "status": "success",
            "message": "Combined auto calibration complete",
            "line_A_y": calib['line_A_y'],
            "line_B_y": calib['line_B_y'],
            "distance_m": calib['calib_distance_m'],
            "road_width_m": calib['road_width_m'],
            "source_points": calib.get('source_points'),
            "calibration_method": calib['calibration_method'],
            "confidence": calib['confidence_score']
        }
        
    except Exception as e:
        calibration_in_progress = False
        print(f"[CombinedCalibration] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500
        
        if tracked_data.get('status') != 'success':
            calibration_in_progress = False
            return {"status": "error", "message": "Tracked vehicle calibration failed", "details": tracked_data}, 500
        
        print(f"✅ Distance estimated: {tracked_data.get('distance_m')}m")
        
        # Update calibration with combined results
        calib['calibration_method'] = 'auto_combined'
        calib['confidence_score'] = 0.85  # High confidence for combined method
        
        print("\n" + "="*60)
        print("✅ COMBINED AUTO CALIBRATION - Complete!")
        print(f"   Lines: A={calib['line_A_y']}, B={calib['line_B_y']}")
        print(f"   Distance: {calib['calib_distance_m']}m")
        print(f"   Road Width: {calib['road_width_m']}m")
        print("="*60 + "\n")
        
        calibration_in_progress = False
        
        return {
            "status": "success",
            "message": "Combined auto calibration complete",
            "line_A_y": calib['line_A_y'],
            "line_B_y": calib['line_B_y'],
            "distance_m": calib['calib_distance_m'],
            "road_width_m": calib['road_width_m'],
            "source_points": calib.get('source_points'),
            "calibration_method": "auto_combined",
            "confidence": 0.85,
            "gemini_details": gemini_data,
            "tracked_details": tracked_data
        }
        
    except Exception as e:
        calibration_in_progress = False
        print(f"[CombinedCalibration] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


# ---------------------------
# Manual Calibration Endpoint
# ---------------------------
@app.route('/api/manual_calibrate', methods=['POST'])
def manual_calibrate():
    """
    Apply manual calibration values from frontend
    """
    global calib
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['line_A_y', 'line_B_y', 'calib_distance_m', 'road_width_m']
        for field in required:
            if field not in data:
                return {"status": "error", "message": f"Missing required field: {field}"}, 400
        
        # Update calibration
        calib['line_A_y'] = int(data['line_A_y'])
        calib['line_B_y'] = int(data['line_B_y'])
        calib['calib_distance_m'] = float(data['calib_distance_m'])
        calib['road_width_m'] = float(data['road_width_m'])
        
        if 'source_points' in data and data['source_points']:
            calib['source_points'] = data['source_points']
        
        calib['calibration_method'] = 'manual'
        calib['confidence_score'] = 1.0
        
        print(f"\n✅ Manual calibration applied:")
        print(f"   Lines: A={calib['line_A_y']}, B={calib['line_B_y']}")
        print(f"   Distance: {calib['calib_distance_m']}m")
        print(f"   Road Width: {calib['road_width_m']}m")
        if calib.get('source_points'):
            print(f"   Polygon: {calib['source_points']}")
        
        # Save calibration to backend API
        if current_camera_id:
            try:
                stats = {
                    "line_ay": int(calib['line_A_y']),
                    "line_by": int(calib['line_B_y']),
                    "polygon_points": calib.get('source_points', []),
                    "distance": float(calib['calib_distance_m']),
                    "width": float(calib['road_width_m']),
                    "method": "manual",
                    "confidence": 1.0,
                    "reason": "Manual calibration"
                }
                
                camera_link = current_video_source
                location = "Unknown"
                
                save_calibration_to_backend(current_camera_id, camera_link, location, stats)
            except Exception as save_error:
                print(f"[BackendAPI] Failed to save calibration: {save_error}")
        
        return {
            "status": "success",
            "message": "Manual calibration applied",
            "calibration": calib
        }
        
    except Exception as e:
        print(f"[ManualCalibration] Error: {e}")
        return {"status": "error", "message": str(e)}, 500


# ---------------------------
# Status Endpoint
# ---------------------------
@app.route('/api/load_calibration/<camera_id>', methods=['GET'])
def load_calibration(camera_id):
    """
    Load calibration data from backend API for a specific camera.
    Also loads speed limit and auto-starts detection if status is ACTIVE.
    """
    global current_camera_id, speed_mode, calib, road_polygon, current_speed_limit
    current_camera_id = camera_id
    
    print(f"[LoadCalibration] Loading data for camera: {camera_id}")
    
    # Fetch calibration data from backend
    backend_data = fetch_calibration_data(camera_id)
    
    # Extract speed limit from calibration data (new API format includes speedLimit)
    speed_limit = None
    if backend_data:
        # Try to get speed limit from response
        speed_limit = backend_data.get('speedLimit')
        if speed_limit:
            print(f"[LoadCalibration] Speed limit from calibration API: {speed_limit} km/h")
        else:
            # Fallback: try separate API call
            speed_limit = fetch_speed_limit(camera_id)
    else:
        # No backend data, try separate API call
        speed_limit = fetch_speed_limit(camera_id)
    
    # Store speed limit globally for violation recorder
    if speed_limit:
        current_speed_limit = speed_limit
    else:
        current_speed_limit = 75  # Final fallback
        print(f"[LoadCalibration] Using default speed limit: {current_speed_limit} km/h")
    
    if backend_data and backend_data.get('stats'):
        stats = backend_data['stats']
        
        # Apply calibration data to local state
        calib['line_A_y'] = stats.get('line_ay', 300)
        calib['line_B_y'] = stats.get('line_by', 500)
        calib['calib_distance_m'] = stats.get('distance', 10.0)
        calib['road_width_m'] = stats.get('width', 22.5)
        calib['calibration_method'] = stats.get('method', 'unknown')
        calib['confidence_score'] = stats.get('confidence', 0.0)
        
        # Load polygon points if available and reconstruct road_polygon
        polygon_points = stats.get('polygon_points')
        if polygon_points and isinstance(polygon_points, list) and len(polygon_points) > 0:
            # Convert to numpy array for road_polygon global (ensure int32 type for OpenCV)
            try:
                road_polygon = np.array(polygon_points, dtype=np.int32)
                calib['source_points'] = polygon_points
                print(f"[LoadCalibration] Loaded and set road_polygon with {len(polygon_points)} points")
                print(f"[LoadCalibration] Road polygon shape: {road_polygon.shape}, dtype: {road_polygon.dtype}")
            except Exception as e:
                print(f"[LoadCalibration] ⚠️ Error converting polygon: {e}")
                road_polygon = None
        else:
            # No polygon available, clear road_polygon to trigger fallback
            road_polygon = None
            print(f"[LoadCalibration] ⚠️ No road polygon available - will use full width fallback")
        
        # DON'T auto-start detection - let user manually start
        current_status = backend_data.get('currentStatus', 'INACTIVE')
        # Note: We load the status but don't auto-start anymore
        # User must manually click "Start Detection" in UI
        print(f"[LoadCalibration] Backend status: {current_status} (manual start required)")
        
        print(f"[LoadCalibration] Calibration loaded successfully")
        print(f"  Line A: {calib['line_A_y']}, Line B: {calib['line_B_y']}")
        print(f"  Distance: {calib['calib_distance_m']}m, Width: {calib['road_width_m']}m")
        print(f"  Speed Limit: {speed_limit} km/h" if speed_limit else "  Speed Limit: Not set")
        print(f"  Status: {current_status}")
        
        return {
            "status": "success",
            "message": "Calibration loaded from backend",
            "calibration": {
                "line_A_y": calib['line_A_y'],
                "line_B_y": calib['line_B_y'],
                "calib_distance_m": calib['calib_distance_m'],
                "road_width_m": calib['road_width_m'],
                "source_points": calib.get('source_points'),
                "method": calib['calibration_method'],
                "confidence": calib['confidence_score']
            },
            "speed_limit": speed_limit,
            "current_status": current_status,
            "auto_started": False  # Never auto-start, user must manually start
        }
    else:
        print(f"[LoadCalibration] No backend data found, using defaults")
        return {
            "status": "success",
            "message": "Using default calibration",
            "calibration": {
                "line_A_y": calib['line_A_y'],
                "line_B_y": calib['line_B_y'],
                "calib_distance_m": calib['calib_distance_m'],
                "road_width_m": calib['road_width_m'],
                "source_points": calib.get('source_points'),
                "method": calib.get('calibration_method', 'unknown'),
                "confidence": calib.get('confidence_score', 0.0)
            },
            "speed_limit": speed_limit,
            "current_status": "INACTIVE",
            "auto_started": False
        }


@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get current system status and calibration values
    """
    with video_cache_lock:
        cache_info = {
            'url': video_cache.get('url'),
            'is_ready': video_cache['is_ready'],
            'is_downloading': video_cache['is_downloading'],
            'download_progress': video_cache['download_progress'],
            'error': video_cache['error']
        }
    
    return {
        "status": "success",
        "line_A_y": calib['line_A_y'],
        "line_B_y": calib['line_B_y'],
        "calib_distance_m": calib['calib_distance_m'],
        "road_width_m": calib['road_width_m'],
        "source_points": calib.get('source_points'),
        "running": calib['running'],
        "calibration_method": calib.get('calibration_method', 'unknown'),
        "confidence_score": calib.get('confidence_score', 0.0),
        "homography_matrix": calib.get('homography_matrix'),
        "video_source": current_video_source,
        "video_cache": cache_info
    }


@app.route('/api/thumbnail', methods=['GET'])
def get_thumbnail():
    """
    Get a thumbnail image from the cached video.
    Returns the first frame as JPEG.
    """
    import io
    from flask import send_file
    
    # Get the video source from query parameter or use cached video
    source = request.args.get('source', '')
    
    if not source:
        # Try to use current cached video
        with video_cache_lock:
            if video_cache.get('file_path') and os.path.exists(video_cache['file_path']):
                source = video_cache['file_path']
            else:
                return {"status": "error", "message": "No video cached"}, 404
    else:
        # Check if this URL has a cached file
        cache_entry = get_cache_entry(source)
        if cache_entry and cache_entry.get('file_path') and os.path.exists(cache_entry['file_path']):
            source = cache_entry['file_path']
        else:
            return {"status": "error", "message": "Video not cached yet"}, 404
    
    try:
        # Open video and extract first frame
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {"status": "error", "message": "Failed to open video"}, 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame"}, 500
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return {"status": "error", "message": "Failed to encode frame"}, 500
        
        # Return as image
        io_buf = io.BytesIO(buffer)
        return send_file(io_buf, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"[Thumbnail] Error: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.route('/api/video_stream', methods=['GET'])
def get_video_stream():
    """
    Stream the cached video file for thumbnail playback.
    Returns the actual video file.
    """
    from flask import send_file
    
    # Get the video source from query parameter
    source = request.args.get('source', '')
    
    if not source:
        return {"status": "error", "message": "Missing source parameter"}, 400
    
    # Check if this URL has a cached file
    cache_entry = get_cache_entry(source)
    if cache_entry and cache_entry.get('file_path') and os.path.exists(cache_entry['file_path']):
        file_path = cache_entry['file_path']
        print(f"[VideoStream] Serving cached video: {file_path}")
        return send_file(file_path, mimetype='video/mp4')
    else:
        return {"status": "error", "message": "Video not cached yet"}, 404


@app.route('/api/set_video_source', methods=['POST'])
def set_video_source():
    """
    Set the video source for calibration and speed detection.
    Accepts either a file path or a stream URL (HTTP, HTTPS, RTSP, etc.)
    """
    global current_video_source
    
    try:
        data = request.get_json(force=True)
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}, 400
    
    source = data.get('source', '').strip()
    if not source:
        return {"status": "error", "message": "Missing 'source' field"}, 400
    
    print(f"[SetVideoSource] Changing video source to: {source}")
    
    # For HTTP streams, check cache first then start downloading/caching if needed
    is_http_stream = source.lower().startswith('http://') or source.lower().startswith('https://')
    
    if is_http_stream:
        # Check if we already have ANY cache (even if stale)
        cache_entry = get_cache_entry(source)
        current_time = time.time()
        
        # Check if cache file exists (regardless of age)
        cache_exists = (cache_entry and 
                       cache_entry.get('file_path') and 
                       os.path.exists(cache_entry['file_path']))
        
        # Check if cache is fresh (within refresh interval)
        cache_is_fresh = (cache_exists and
                         (current_time - cache_entry.get('download_time', 0)) < CACHE_REFRESH_INTERVAL)
        
        # Check if download is already in progress
        with video_cache_lock:
            is_already_downloading = video_cache.get('is_downloading', False)
        
        if cache_exists:
            cache_age_min = int((current_time - cache_entry.get('download_time', 0)) / 60)
            print(f"[SetVideoSource] Using existing cache (age: {cache_age_min} min)")
            
            # Always use existing cache immediately
            with video_cache_lock:
                video_cache['url'] = source
                video_cache['file_path'] = cache_entry['file_path']
                video_cache['download_time'] = cache_entry['download_time']
                video_cache['is_ready'] = True
            
            # If cache is stale and no download in progress, refresh in background
            if not cache_is_fresh and not is_already_downloading:
                print(f"[SetVideoSource] Cache is stale, refreshing in background...")
                def download_in_background():
                    cached_file = download_and_cache_stream(source, force_refresh=True)
                    if cached_file:
                        print(f"[SetVideoSource] Background refresh complete")
                
                import threading
                download_thread = threading.Thread(target=download_in_background, daemon=True)
                download_thread.start()
            elif not cache_is_fresh:
                print(f"[SetVideoSource] Refresh already in progress")
        elif is_already_downloading:
            print(f"[SetVideoSource] Download already in progress, waiting...")
        else:
            # No cache exists, download for the first time
            print(f"[SetVideoSource] No cache found, downloading...")
            def download_in_background():
                cached_file = download_and_cache_stream(source)
                if cached_file:
                    print(f"[SetVideoSource] Initial download complete")
                    start_cache_refresh_thread(source)
            
            import threading
            download_thread = threading.Thread(target=download_in_background, daemon=True)
            download_thread.start()
    
    current_video_source = source
    
    return {
        "status": "success",
        "message": f"Video source set to: {source}",
        "source": current_video_source
    }


# ---------------------------
# Cameras Proxy Endpoint
# ---------------------------
@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    """Fetch camera list from external service server-side to avoid browser CORS."""
    # Add cache-busting query to avoid any intermediary caching at the provider
    import time as _time
    url = f'https://nextgen-fv1h.onrender.com/api/calibhome?_t={int(_time.time())}'
    try:
        import urllib.request
        import json
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            try:
                payload = json.loads(raw.decode('utf-8'))
            except Exception:
                payload = []
        # Normalize common shapes
        if isinstance(payload, dict):
            # Typical wrappers
            for key in ('data', 'cameras', 'items', 'results'): 
                if key in payload and isinstance(payload[key], list):
                    payload = payload[key]
                    break
            else:
                # Heuristic: find first list of dicts with cameraId/cameraLink keys
                candidate = None
                for v in payload.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        keys = set(v[0].keys())
                        if {'cameraId', 'cameraLink'} & keys:
                            candidate = v
                            break
                payload = candidate if candidate is not None else []
        # Ensure a list is returned for frontend mapping
        if isinstance(payload, list):
            return payload
        return []
    except Exception as e:
        print(f"[CamerasProxy] Error fetching cameras: {e}")
        return {"status": "error", "message": str(e)}, 502


@app.route('/api/calibhome', methods=['POST'])
def api_calibhome():
    """Proxy POST to external calibhome API to add a camera (avoids browser CORS)."""
    import json
    from flask import jsonify
    try:
        data = request.get_json(force=True)
    except Exception:
        data = None
    if not data or not isinstance(data, dict):
        return {"status": "error", "message": "Invalid JSON body"}, 400

    # Validate expected fields
    for k in ("cameraId", "cameraLink", "location"):
        if k not in data or not str(data[k]).strip():
            return {"status": "error", "message": f"Missing field: {k}"}, 400

    url = 'https://nextgen-fv1h.onrender.com/api/calibhome'
    try:
        import urllib.request
        payload = json.dumps({
            "cameraId": str(data["cameraId"]).strip(),
            "cameraLink": str(data["cameraLink"]).strip(),
            "location": str(data["location"]).strip(),
        }).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'User-Agent': 'Mozilla/5.0',
                'Content-Type': 'application/json'
            },
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            raw = resp.read()
            try:
                result = json.loads(raw.decode('utf-8'))
            except Exception:
                result = {"status": "ok" if 200 <= status < 300 else "error"}
        # Log upstream response in proper JSON (double quotes)
        try:
            import json as _json
            print(f"[CalibHomeProxy] Upstream status {status}, response: {_json.dumps(result, ensure_ascii=False)}")
        except Exception:
            try:
                print(f"[CalibHomeProxy] Upstream status {status}, response: {result}")
            except Exception:
                pass
        return (result, status)
    except Exception as e:
        print(f"[CalibHomeProxy] Error posting camera: {e}")
        return {"status": "error", "message": str(e)}, 502


@app.route('/api/calibhome/<camera_id>', methods=['DELETE'])
def api_calibhome_delete(camera_id):
    """Proxy DELETE to external calibhome API to remove a camera (avoids browser CORS)."""
    import json
    if not camera_id or not camera_id.strip():
        return {"status": "error", "message": "Missing camera ID"}, 400

    url = f'https://nextgen-fv1h.onrender.com/api/calibhome/{camera_id.strip()}'
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'},
            method='DELETE'
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            raw = resp.read()
            try:
                result = json.loads(raw.decode('utf-8'))
            except Exception:
                result = {"status": "ok" if 200 <= status < 300 else "error"}
        # Log upstream response
        try:
            import json as _json
            print(f"[CalibHomeProxy] DELETE {camera_id} - status {status}, response: {_json.dumps(result, ensure_ascii=False)}")
        except Exception:
            try:
                print(f"[CalibHomeProxy] DELETE {camera_id} - status {status}, response: {result}")
            except Exception:
                pass
        return (result, status)
    except Exception as e:
        print(f"[CalibHomeProxy] Error deleting camera {camera_id}: {e}")
        return {"status": "error", "message": str(e)}, 502


# ---------------------------
# Start Speed Detection
# ---------------------------
@app.route('/api/start_speed', methods=['POST'])
def api_start_speed():
    """Start speed detection and update backend status"""
    global calib, speed_mode, speed_buffer, track_state, violation_recorder, current_speed_limit
    # Reset buffers/state for a clean start
    speed_buffer.clear()
    track_state = {}
    speed_mode = True
    calib['running'] = True
    print("✅ Speed detection started via API")
    
    # 🔹 NEW: Initialize violation recorder with stored speed limit and calibration lines
    with violation_recorder_lock:
        # Use the globally stored speed limit from calibration
        speed_limit = current_speed_limit
        
        # Get calibration line coordinates
        line_a_y = calib.get('line_A_y')
        line_b_y = calib.get('line_B_y')
        
        violation_recorder = create_violation_recorder(
            camera_id=current_camera_id or "UNKNOWN",
            speed_limit=speed_limit,
            line_a_y=line_a_y,
            line_b_y=line_b_y
        )
        print(f"[ViolationRecorder] Initialized with speed limit: {speed_limit} km/h")
        if line_a_y is not None and line_b_y is not None:
            print(f"[ViolationRecorder] Middle line Y: {(line_a_y + line_b_y) / 2}")
    
    # Update backend status to ACTIVE
    if current_camera_id:
        update_camera_status(current_camera_id, "ACTIVE")
    
    return {"status": "success", "message": "Speed detection started"}


# ---------------------------
# Stop Speed Detection
# ---------------------------
@app.route('/api/stop_speed', methods=['POST'])
def api_stop_speed():
    """Stop speed detection and update backend status"""
    global calib, speed_mode, violation_recorder
    speed_mode = False
    calib['running'] = False
    print("⏹ Speed detection stopped via API")
    
    # 🔹 NEW: Cleanup violation recorder
    with violation_recorder_lock:
        if violation_recorder is not None:
            violation_recorder.cleanup()
            violation_recorder = None
            print("[ViolationRecorder] Cleaned up")
    
    # Update backend status to INACTIVE
    if current_camera_id:
        update_camera_status(current_camera_id, "INACTIVE")
    
    return {"status": "success", "message": "Speed detection stopped"}


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    # Initialize cache system
    print("[VideoCache] Initializing cache system...")
    ensure_temp_dir()
    
    # Load existing cache metadata
    metadata = load_cache_metadata()
    if metadata:
        print(f"[VideoCache] Found {len(metadata)} cached streams in metadata")
        
        # Verify files exist and restore most recent valid cache to memory
        valid_entries = []
        for url, entry in metadata.items():
            if os.path.exists(entry['file_path']):
                valid_entries.append(entry)
                # Start auto-refresh thread for existing cache
                print(f"[VideoCache] Starting auto-refresh for cached stream: {url}")
                start_cache_refresh_thread(url)
            else:
                print(f"[VideoCache] Stale cache entry (file missing): {entry['file_path']}")
        
        # Restore most recent cache to in-memory state if available
        if valid_entries:
            most_recent = max(valid_entries, key=lambda e: e.get('download_time', 0))
            with video_cache_lock:
                video_cache['url'] = most_recent['url']
                video_cache['file_path'] = most_recent['file_path']
                video_cache['download_time'] = most_recent['download_time']
                video_cache['is_ready'] = True
            print(f"[VideoCache] Restored most recent cache: {most_recent['url']}")
    
    # Cleanup orphaned files
    cleanup_old_caches()
    
    print("[VideoCache] Cache system initialized")
    
    # Start global cleanup thread for temp files
    start_global_cleanup_thread()
    
    # Check if there's an active camera and auto-start detection
    # This runs in a separate thread to not block app startup
    def check_and_autostart_detection():
        time.sleep(3)  # Wait for app to fully initialize
        
        # Try to find last active camera from backend
        # For now, we'll check if there's a recent calibration with ACTIVE status
        print("[AutoStart] Checking for active camera status...")
        
        # You could extend this to check a specific camera ID or all cameras
        # For now, this will be triggered when frontend loads calibration
        pass
    
    autostart_thread = threading.Thread(target=check_and_autostart_detection, daemon=True)
    autostart_thread.start()
    
    # Start frame processor thread
    t = threading.Thread(target=frame_processor, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5001, debug=False)


