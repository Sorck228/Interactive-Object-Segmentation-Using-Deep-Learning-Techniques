from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from model_utils import get_loaders

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 0
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../../Github_data/data/train_images/"
TRAIN_MASK_DIR = "../../Github_data/data/train_masks/"
VAL_IMG_DIR = "../../Github_data/data/test_images/"
VAL_MASK_DIR = "../../Github_data/data/test_masks/"

# Working but not useful for this application
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x

class firstModel(nn.Module):  #  Work in progress
    def __init__(self):
        super(firstModel, self).__init__()
        model = models.resnet50(pretrained=True)
        for child in model.children():
            self.tester = child[4]
            print(self.tester)
            self.Conv1 = child[0]   #
            self.BN1 = child[1]     #
            # child[2] = ReLU   #
            self.Conv4 = child[7]   #
            self.Conv5 = child[10]  #
            self.Conv6 = child[12]  #
            self.Conv7 = child[14]  #
            self.upSample1 = nn.Upsample(scale_factor=2)
            self.upSample2 = nn.Upsample(scale_factor=4)
            break

    def forward(self, x):
        out1 = self.Conv1(x)
        out1 = nn.ReLU(out1)
        out1 = self.Conv2(out1)
        out1 = nn.ReLU(out1)
        out1_mp = nn.MaxPool2d(out1, 2, 2)
        out2 = self.Conv3(out1_mp)
        out2 = nn.ReLU(out2)
        out2 = self.Conv4(out2)
        out2 = nn.ReLU(out2)
        out2_mp = nn.MaxPool2d(out2, 2, 2)
        out3 = self.Conv5(out2_mp)
        out3 = nn.ReLU(out3)
        out3 = self.Conv6(out3)
        out3 = nn.ReLU(out3)
        out3 = self.Conv7(out3)
        out3 = nn.ReLU(out3)
        ###### up sampling to create output with the same size
        out2 = self.upSample1(out2)
        out3 = self.upSample2(out3)
        #out7_mp = F.max_pool2d(out7, 2, 2)
        concat_features = torch.cat([out1, out2, out3], 1)
        return out1, concat_features


resnet50 = models.resnet50(pretrained=True).eval()
img = Image.open("dog.jpeg")


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


child_counter = 0
for child in resnet50.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

# plt.imshow(img); plt.show()

# define transformations
trf = A.Compose([A.Resize(height=256, width=256),
                 A.CenterCrop(height=224, width=224),
                 A.pytorch.ToTensorV2(),
                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# removing avg. pool & FCN layers for both streams
image_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
click_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))

# freeze resnet layer weights
for param in image_block.parameters():
    param.requires_grad = False

for param in click_block.parameters():
    param.requires_grad = False


use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if use_gpu:
    print("Using CUDA")
image_block = image_block.cuda() if use_gpu else image_block
click_block = click_block.cuda() if use_gpu else click_block

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(image_block.parameters(), lr=0.0001, momentum=0.9)


# num_ftrs = image_block.fc.in_features
# image_block.fc = nn.Linear(num_ftrs, 128)
# image_block.fc = image_block.fc.cuda() if use_gpu else image_block.fc



train_dataloader, test_dataloader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        trf,
        trf,
        NUM_WORKERS,
        PIN_MEMORY,
    )


print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
print(total_step)


