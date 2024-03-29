from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from model_utils import get_loaders_COCO
import sys

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 0
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../../Github_data/coco/img_reduced_sample/"
TRAIN_MASK_DIR = "../../Github_data/coco/mask_reduced_sample/"
VAL_IMG_DIR = "../../Github_data/data/test_images/"
VAL_MASK_DIR = "../../Github_data/data/test_masks/"

def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


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

img = Image.open("dog.jpeg")
img = np.array(img)

coco_mask = Image.open("../../Github_data/coco/mask_reduced_sample/000000022704.png")
coco_mask = np.array(coco_mask)



resnet50 = models.resnet50(pretrained=True).eval()
resnet50_fcn = models.segmentation.fcn_resnet50(pretrained=True).eval()

#
# child_counter = 0
# for child in resnet50_fcn.classifier.children():
#    print(" child", child_counter, "is:")
#    print(child)
#    child_counter += 1

# plt.imshow(img); plt.show()


# removing avg. pool & FCN layers for both streams
image_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
click_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))

decoder_block_ = torch.nn.Sequential(*(list(resnet50_fcn.classifier.children())[:]))
decoder_block = torch.nn.Sequential(decoder_block_,
                                    nn.Conv2d(21, 2, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Softmax(),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.UpsamplingBilinear2d(scale_factor=2)
                                    )



if DEVICE:
    print("Using CUDA")
image_block = image_block.cuda() if DEVICE=="cuda" else image_block
click_block = click_block.cuda() if DEVICE=="cuda" else click_block
decoder_block = image_block.cuda() if DEVICE=="cuda" else decoder_block

# freeze resnet layer weights
for param in image_block.parameters():
    param.requires_grad = False

for param in click_block.parameters():
    param.requires_grad = False

train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )
test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(image_block.parameters(), lr=0.0001, momentum=0.9)


train_dataloader, test_dataloader = get_loaders_COCO(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

sample = False
if sample:
    sample_idx = torch.randint(len(train_dataloader.dataset), size=(1,)).item()
    img, label = train_dataloader.dataset[sample_idx]

    plt.figure("image")
    plt.axis("off")
    plt.imshow(img.T)
    plt.show()

    plt.figure("label")
    plt.axis("off")
    plt.imshow(label.T)
    plt.show()


print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader.dataset.images)
print(f"total step: {total_step}")


for epoch in range(1, NUM_EPOCHS+1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # add batch dimension to target and send to device
        # format  [batch_size, channel, height, width]
        #data_, target_ = data.to(DEVICE), targets.float().unsqueeze(1).to(device=DEVICE)
        data_, target_ = data.to(DEVICE), targets.long().to(device=DEVICE)

        # reset optimazer gradients
        optimizer.zero_grad()

        # run model
        out_image_block = image_block(data_)
        out_decoder_block = decoder_block(out_image_block)

        # debugging step to see model output and target shapes
        print("target shape {}".format(target_.shape))
        print("output shape {}".format(out_decoder_block.shape))

        loss = criterion(out_decoder_block, target_)
        loss.backward()

        optimizer.step()

#         running_loss += loss.item()
#         _, pred = torch.max(out_decoder_block, dim=1)
#         correct += torch.sum(pred == target_).item()
#         total += target_.size(0)
#         if (batch_idx) % 20 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, NUM_EPOCHS, batch_idx, total_step, loss.item()))
# #
#         # forward
#
#
#         # backward

    if epoch == NUM_EPOCHS:
        print("Done")



######################################
# print image_block model layers and test output shape from model, should be 7x7
#child_counter = 0
#for child in image_block.children():
   #print(" child", child_counter, "is:")
   #print(child)
#   child_counter += 1

# transform image and add batch dimension
# transformed_img = test_transforms(image=img)["image"].unsqueeze(0)
#
# print(transformed_img.shape)
# out2 = image_block(transformed_img)
# print(out2.shape)
