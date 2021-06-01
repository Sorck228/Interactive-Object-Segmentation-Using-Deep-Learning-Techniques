import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 20
NUM_EPOCHS = 100
NUM_WORKERS = 6
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.RandomCrop(height=150, width=150),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.Blur(p=0.3),
        # A.CLAHE(p=0.3),
        # A.ColorJitter(p=0.3),
        # A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=728, width=728),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

