"""
Direct dataset download without Kaggle API
Using GDrive or direct URL
"""

import os
import requests
import zipfile
from pathlib import Path
import urllib.request

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"📥 Downloading {filename}...")
    
    # Simple download without progress bar
    urllib.request.urlretrieve(url, filename)
    print(f"✓ Downloaded to {filename}")
    return filename

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"📂 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")

def download_plantvillage_from_gdrive():
    """
    Download PlantVillage dataset from Google Drive
    हा dataset small आहे (पण काम करेल)
    """
    
    # हा सिंगापोर University चा mirror link आहे
    # Contains 3 classes (Apple Black Rot, Apple Healthy, Grape Healthy)
    url = "https://github.com/christianversloot/machine-learning-articles/raw/main/data/plantvillage_kaggle_small.zip"
    
    zip_path = DATA_DIR / "plantvillage_small.zip"
    
    try:
        # Download
        download_file(url, zip_path)
        
        # Extract
        extract_zip(zip_path, DATA_DIR)
        
        print("\n✅ Dataset downloaded successfully!")
        print(f"📁 Location: {DATA_DIR}")
        
        # Show folder structure
        print("\n📁 Folder structure:")
        for item in DATA_DIR.iterdir():
            if item.is_dir():
                print(f"   - {item.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTrying alternative method...")
        return False

def download_via_manual():
    """
    Manual download instructions
    """
    print("\n" + "="*50)
    print("MANUAL DOWNLOAD METHOD (Recommended)")
    print("="*50)
    
    print("""
Step 1: तुझा browser मध्ये ही link उघड:
🔗 https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Step 2: "Download" बटन वर क्लिक कर (शक्यतो तूसुद्धा login झालेला हवेस)

Step 3: Download झाल्यावर ती zip file extract कर

Step 4: Extracted folder ला खालील path वर ठेव:
    """ + str(DATA_DIR) + """

Step 5: Folder structure अशी असायला हवी:
    data/raw/PlantVillage/
        ├── Apple___Black_rot/
        ├── Apple___healthy/
        ├── Grape___healthy/
        └── ... (other classes)

अथवा छोट्या dataset साठी खालील link वापर (38 MB):
🔗 https://github.com/christianversloot/machine-learning-articles/raw/main/data/plantvillage_kaggle_small.zip
    """)
    
    input("\nPress Enter after you have placed the dataset...")
    
    # Check if dataset exists
    if check_dataset_exists():
        print("✅ Dataset found!")
        return True
    else:
        print("❌ Dataset not found. Please check the path.")
        return False

def check_dataset_exists():
    """Check if dataset is present"""
    expected_path = DATA_DIR / "PlantVillage"
    
    if expected_path.exists():
        num_classes = len([d for d in expected_path.iterdir() if d.is_dir()])
        print(f"✓ Found dataset with {num_classes} classes")
        return True
    
    # Check for extracted folder
    for item in DATA_DIR.iterdir():
        if item.is_dir() and any(item.iterdir()):
            print(f"✓ Found dataset folder: {item.name}")
            return True
    
    return False

def main():
    print("="*50)
    print("PLANT DISEASE DATASET DOWNLOADER")
    print("="*50)
    
    # First try automatic download
    success = download_plantvillage_from_gdrive()
    
    if not success:
        # If fails, give manual instructions
        download_via_manual()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("1. Run: python src/data_preprocessing.py")
    print("2. Run: python src/train.py")
    print("="*50)

if __name__ == "__main__":
    main()