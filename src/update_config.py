import os

config_path = 'src/config.py'

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        content = f.read()
    
    # EPOCHS update
    if 'EPOCHS = 20' in content:
        content = content.replace('EPOCHS = 20', 'EPOCHS = 2')
    if 'EPOCHS = 3' in content:
        content = content.replace('EPOCHS = 3', 'EPOCHS = 2')
    
    with open(config_path, 'w') as f:
        f.write(content)
    print('✅ Updated EPOCHS to 2 for quick training')
else:
    print('⚠️ config.py not found, skipping')