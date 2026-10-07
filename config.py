import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = Path("DATA/INPUTS/sample3.mp4")
MODEL_PATH = Path("yolov8n.pt")
CONF = 0.30
INFERENCE_SIZE = 416
DETECTION_STRIDE = 3
CLASSES = {2: "car", 3: "bike", 5: "bus", 7: "truck"}

# Starter calibration spans the sample frame. Run calibrate.py for road-only points.
SOURCE_POINTS = [[0, 0], [1363, 0], [1363, 765], [0, 765]]
REAL_WIDTH_M = 7.0
REAL_LENGTH_M = 20.0
SPEED_LIMIT = 60
SPEED_WINDOW = 10
MIN_SPEED_SAMPLES = 5
MAX_VALID_SPEED = 250.0
PORT = int(os.environ.get("TRAFFIC_PORT", "5000"))

INPUT_DIR = PROJECT_ROOT / "DATA" / "INPUTS"
OUTPUT_DIR = PROJECT_ROOT / "DATA" / "OUTPUTS"
VEHICLES_LOG = OUTPUT_DIR / "vehicles_log.csv"
OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"
RESULT_METADATA = OUTPUT_DIR / "result_metadata.json"
