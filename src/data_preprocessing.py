"""
Data Preprocessing - Handles Augmented Dataset with Subfolders
"""

import sys
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import *

def collect_all_images(root_dir):
    """
    Recursively collect all images from all subfolders
    """
    images = []
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    
    # Walk through all subdirectories
    for ext in extensions:
        for img_path in root_dir.glob(f'**/{ext}'):
            # Class name is the parent of the parent (Common_rust is grandparent)
            grandparent = img_path.parent.parent
            class_name = grandparent.name
            
            # Clean up class name
            if 'Common_rust' in class_name:
                class_name = 'Common_rust'
            elif 'Northern_Leaf_Blight' in class_name:
                class_name = 'Northern_Leaf_Blight'
            elif 'healthy' in class_name:
                class_name = 'healthy'
            elif 'Cercospora' in class_name or 'leaf_spot' in class_name:
                class_name = 'Cercospora_leaf_spot'
            
            images.append({
                'path': img_path,
                'class': class_name
            })
    
    return images

def organize_train_val_test():
    """
    Organize all images into train/val/test folders
    """
    corn_dir = RAW_DATA_DIR / "Corn"
    
    print(f"📁 Scanning: {corn_dir}")
    print("🔄 Collecting images from all augmented subfolders...\n")
    
    # Collect all images
    all_images = collect_all_images(corn_dir)
    
    if len(all_images) == 0:
        print("❌ No images found!")
        return None, None, None
    
    print(f"✅ Found {len(all_images)} total images\n")
    
    # Count images per class
    class_counts = {}
    for img in all_images:
        class_counts[img['class']] = class_counts.get(img['class'], 0) + 1
    
    print("📊 Class distribution:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count:,} images")
    
    # Clear processed directory
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    # Create train/val/test directories
    for split in ['train', 'val', 'test']:
        for class_name in class_counts.keys():
            (PROCESSED_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Split and copy images class-wise
    for class_name in class_counts.keys():
        class_images = [img for img in all_images if img['class'] == class_name]
        
        # Split: 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(class_images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        print(f"\n📂 {class_name}:")
        print(f"   Train: {len(train_imgs):,} images")
        print(f"   Val: {len(val_imgs):,} images")
        print(f"   Test: {len(test_imgs):,} images")
        
        # Copy images
        for img in train_imgs:
            shutil.copy2(img['path'], PROCESSED_DATA_DIR / "train" / class_name / f"{img['path'].parent.name}_{img['path'].name}")
        for img in val_imgs:
            shutil.copy2(img['path'], PROCESSED_DATA_DIR / "val" / class_name / f"{img['path'].parent.name}_{img['path'].name}")
        for img in test_imgs:
            shutil.copy2(img['path'], PROCESSED_DATA_DIR / "test" / class_name / f"{img['path'].parent.name}_{img['path'].name}")
    
    print("\n" + "="*50)
    print("✅ Data organization complete!")
    print("="*50)
    
    return create_data_generators()

def create_data_generators():
    """
    Create TensorFlow data generators with augmentation
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print("\n📊 Creating data generators...")
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation/Test generator (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / "train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / "val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / "test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✅ Generators ready!")
    print(f"   Training batches: {len(train_generator)}")
    print(f"   Validation batches: {len(val_generator)}")
    print(f"   Test batches: {len(test_generator)}")
    
    return train_generator, val_generator, test_generator

def main():
    print("="*60)
    print("🌽 CORN LEAF DISEASE - DATA PREPROCESSING")
    print("="*60)
    
    train_gen, val_gen, test_gen = organize_train_val_test()
    
    if train_gen and train_gen.samples > 0:
        print("\n" + "="*60)
        print("🎉 DATA PREPROCESSING SUCCESSFUL!")
        print("="*60)
        print(f"\n📁 Processed data: {PROCESSED_DATA_DIR}")
        print(f"📊 Total training samples: {train_gen.samples:,}")
        print(f"\n🚀 Next step: Run 'python src/train.py'")
    else:
        print("\n❌ Something went wrong. Please check the errors above.")

if __name__ == "__main__":
    main()