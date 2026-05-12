"""
Predict disease from a single image
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import CLASS_NAMES, IMAGE_SIZE

def predict_image(image_path, model_path=None):
    """Predict disease from a single image"""
    
    # Load model
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "best_model.keras"
    
    model = load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Display results
    print("\n" + "="*40)
    print("PREDICTION RESULT")
    print("="*40)
    print(f"Image: {image_path.name}")
    print(f"Predicted Disease: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("="*40)
    
    # Show all class probabilities
    print("\nAll class probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {predictions[0][i]:.2%}")
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'all_probabilities': predictions[0]
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        predict_image(image_path)
    else:
        print("Usage: python predict.py path/to/image.jpg")