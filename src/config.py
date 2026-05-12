"""
Configuration file for Plant Disease Classification Project
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Create directories if not exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, DASHBOARD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset parameters
IMAGE_SIZE = (224, 224)  # MobileNetV2 expects 224x224
BATCH_SIZE = 64
NUM_CLASSES = 4  # Corn dataset has 4 classes

# Class names (Corn dataset)
CLASS_NAMES = [
    'Cercospora_leaf_spot',      # Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot
    'Common_rust',                # Corn_(maize)___Common_rust_
    'Northern_Leaf_Blight',       # Corn_(maize)___Northern_Leaf_Blight
    'healthy'                     # Corn_(maize)___healthy
]

# Training parameters
EPOCHS = 3
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data augmentation parameters
AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}