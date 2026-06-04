import sys
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image

# ==========================================
# Configuration
# ==========================================
MODEL_PATH = 'best_galaxy_swint.pth'
IMAGE_SIZE = 224
TTA_N      = 16

CLASS_NAMES = [
    'Barred_Spiral', 'Cigar_Smooth', 'Disturbed', 'Edge_On_Bulge',
    'Edge_On_No_Bulge', 'Inbetween_Smooth', 'Merging', 'Round_Smooth',
    'Unbarred_Loose_Spiral', 'Unbarred_Tight_Spiral'
]
NUM_CLASSES = len(CLASS_NAMES)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = 'cuda' if USE_AMP else 'cpu'

# ==========================================
# Load Model
# ==========================================
SWIN_T_FEATURES = 768

model = models.swin_t(weights=None)
model.head = nn.Sequential(
    nn.Linear(SWIN_T_FEATURES, 512),
    nn.BatchNorm1d(512),
    nn.GELU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ==========================================
# Transforms
# ==========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Converts PIL → (C, H, W) float32 tensor in [0, 1], no normalization yet.
# Augmentation must happen before normalization so rotation fill-value (0.0)
# corresponds to black, matching how the model was trained.
base_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Spatial augmentation only — applied on [0, 1] float tensor BEFORE normalize
tta_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomChoice([
        v2.RandomRotation((0,   0)),
        v2.RandomRotation((90,  90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270))
    ]),
])

# ==========================================
# Predict
# ==========================================
def predict(image_path):
    img    = Image.open(image_path).convert('RGB')
    tensor = base_transform(img).to(device)   # (C, H, W), [0, 1], not yet normalized

    accumulated = None
    with torch.no_grad(), torch.autocast(device_type=AMP_DTYPE, enabled=USE_AMP):
        for _ in range(TTA_N):
            # FIX: augment on [0, 1] values first, then normalize, then batch
            aug   = normalize(tta_transform(tensor)).unsqueeze(0)   # (1, C, H, W)
            probs = torch.softmax(model(aug), dim=1)
            accumulated = probs if accumulated is None else accumulated + probs

    avg_probs            = accumulated / TTA_N
    confidence, pred_idx = avg_probs.squeeze().max(dim=0)

    print(f"Image     : {image_path}")
    print(f"Prediction: {CLASS_NAMES[pred_idx.item()]}")
    print(f"Confidence: {confidence.item() * 100:.1f}%\n")
    print("All class probabilities:")
    for name, prob in zip(CLASS_NAMES, avg_probs.squeeze().tolist()):
        bar = '█' * int(prob * 40)
        print(f"  {name:<25} {prob * 100:5.1f}%  {bar}")

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to image: ").strip()
    predict(image_path)
