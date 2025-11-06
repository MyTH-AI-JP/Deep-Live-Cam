# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
souce_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use souce_target_map or simple_map for specific swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')
ui_ready: bool = False               # True after UI widgets are initialized

# Detection / Mapping thresholds
det_thresh: float = 0.5               # InsightFace detection threshold (higher = stricter)
assign_min_similarity: float = 0.6    # Min cosine similarity to assign face to a cluster

# Clustering configuration
cluster_method: str = "elbow"         # 'elbow' or 'silhouette'
cluster_max_k: int = 10               # Upper bound for K search
cluster_min_cluster_size: int = 1     # Minimum total faces assigned across frames to keep a cluster

# Nano Banana (Gemini) preprocessing
enable_nano_banana: bool = True
nano_banana_user_override: bool = False  # True if user explicitly set enable/disable via CLI
nano_banana_on_target: bool = False   # Apply to target image (videos are skipped)
nano_banana_model: str = "gemini-2.5-flash-image"
nano_banana_prompt: str = (
    "Remove all the bangs, fringe, and sideburns, caps, hats, and any kind of headwear so the forehead is fully visible while keeping the rest of the image unchanged. "
    "Preserve lighting, color tone, background, identity, and facial features clearly and in detail. Perform natural inpainting."
)

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---