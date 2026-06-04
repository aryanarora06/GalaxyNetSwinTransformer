import h5py, os
import numpy as np
from PIL import Image

H5_PATH = r'C:\Users\DELL\Desktop\GalaxyNet\Galaxy10_DECals.h5'
OUT_DIR = r'C:\Users\DELL\Desktop\GalaxyNet\data'
CLASS_NAMES = [
    'Disturbed', 'Merging', 'Round_Smooth', 'Inbetween_Smooth',
    'Cigar_Smooth', 'Barred_Spiral', 'Unbarred_Tight_Spiral',
    'Unbarred_Loose_Spiral', 'Edge_On_No_Bulge', 'Edge_On_Bulge'
]

with h5py.File(H5_PATH, 'r') as f:
    images = f['images'][:]   # (17736, 256, 256, 3), uint8
    labels = f['ans'][:]      # (17736,), int

for cls_id, cls_name in enumerate(CLASS_NAMES):
    os.makedirs(os.path.join(OUT_DIR, cls_name), exist_ok=True)

for i, (img, lbl) in enumerate(zip(images, labels)):
    cls_name = CLASS_NAMES[int(lbl)]
    path     = os.path.join(OUT_DIR, cls_name, f'{i:05d}.png')
    Image.fromarray(img).save(path)

print("Done")