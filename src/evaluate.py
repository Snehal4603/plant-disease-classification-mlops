"""
Step 5: Model Evaluation
Confusion matrix, classification report, metrics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_preprocessing import create_data_generators

def evaluate_model():
    """Evaluate trained model on test data"""
    print("="*50)
    print("STEP 5: MODEL EVALUATION")
    print("="*50)
    
    # Load data
    _, _, test_generator = create_data_generators()
    
    # Load best model
    model_path = MODEL_DIR / "best_model.keras"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run train.py first!")
        return
    
    model = load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
    
    print(f"\n📈 Test Metrics:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    
    # Predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Corn Disease Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.show()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n📋 Classification Report:")
    print(report)
    
    # Save report
    with open(MODEL_DIR / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall
    }])
    metrics_df.to_csv(MODEL_DIR / 'test_metrics.csv', index=False)
    
    # Save individual predictions for dashboard
    results_df = pd.DataFrame({
        'filename': test_generator.filenames,
        'true_class': [class_names[i] for i in y_true],
        'predicted_class': [class_names[i] for i in y_pred],
        'correct': y_true == y_pred
    })
    results_df.to_csv(MODEL_DIR / 'predictions.csv', index=False)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved to {MODEL_DIR}")
    print("   - confusion_matrix.png")
    print("   - classification_report.txt")
    print("   - test_metrics.csv")
    print("   - predictions.csv")

if __name__ == "__main__":
    evaluate_model()