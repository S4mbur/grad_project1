from PIL import Image
import os

tile_dir = 'data/tiles'
count = 0
for root, dirs, files in os.walk(tile_dir):
    for f in files:
        if f.endswith('.jpg') or f.endswith('.png'):
            path = os.path.join(root, f)
            img = Image.open(path)
            print(f'{path}: {img.size}')
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break
