import numpy as np
from PIL import Image
from pathlib import Path

base_dir = Path('data/raw/Corn_synthetic')
classes = ['Common_rust', 'Cercospora_leaf_spot', 'Northern_Leaf_Blight', 'healthy']

for class_name in classes:
    class_dir = base_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(50):
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(class_dir / f'{class_name}_{i}.jpg')

print('✅ Synthetic dataset created with 200 images (50 per class)')