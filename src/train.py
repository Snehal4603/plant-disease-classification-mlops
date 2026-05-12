"""
Step 4: Train CNN Model using Transfer Learning (MobileNetV2)
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import *
from src.data_preprocessing import create_data_generators

# MLflow (optional - install if needed)
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠ MLflow not installed. Skipping experiment tracking.")

class PlantDiseaseClassifier:
    def __init__(self, input_shape=(224,224,3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build MobileNetV2 based transfer learning model"""
        print("\n🔨 Building model...")
        
        # Load pre-trained MobileNetV2 (without top layers)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers (initially)
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("✓ Model built successfully!")
        self.model.summary()
        return model
    
    def train(self, train_generator, val_generator, epochs=EPOCHS):
        """Train the model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(MODEL_DIR / "best_model.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(LOG_DIR / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S"))
            )
        ]
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file:{LOG_DIR / 'mlruns'}")
            mlflow.set_experiment("corn_disease_classification")
            mlflow.tensorflow.autolog()
        
        # Train
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(MODEL_DIR / "final_model.keras")
        
        # Save training history
        with open(MODEL_DIR / "training_history.pkl", 'wb') as f:
            pickle.dump(self.history.history, f)
        
        # Plot and save training curves
        self.plot_training_history()
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self.history
    
    def plot_training_history(self):
        """Plot accuracy and loss curves"""
        if not self.history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'training_history.png', dpi=150)
        plt.show()
        print(f"✓ Training plot saved to {MODEL_DIR / 'training_history.png'}")

def main():
    """Main training function"""
    print("="*50)
    print("STEP 4: MODEL TRAINING")
    print("="*50)
    
    # Load data
    print("\n📊 Loading data...")
    train_generator, val_generator, test_generator = create_data_generators()
    
    if train_generator is None:
        print("❌ Failed to load data. Please check data preprocessing.")
        return
    
    # Update num_classes based on actual data
    actual_num_classes = len(train_generator.class_indices)
    print(f"\n📈 Number of classes: {actual_num_classes}")
    print(f"   Classes: {train_generator.class_indices}")
    
    # Create and train model
    classifier = PlantDiseaseClassifier(num_classes=actual_num_classes)
    classifier.build_model()
    classifier.train(train_generator, val_generator)
    
    print("\n" + "="*50)
    print("✓ MODEL TRAINING COMPLETE!")
    print("="*50)
    print(f"\n📁 Model saved at: {MODEL_DIR}")
    print(f"📊 Run: tensorboard --logdir {LOG_DIR / 'tensorboard'}")

if __name__ == "__main__":
    main()