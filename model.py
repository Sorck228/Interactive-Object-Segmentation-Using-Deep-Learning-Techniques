from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
from albumentations.pytorch import ToTensorV2

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

# Working but not useful for this application
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x

class firstModel(nn.Module):
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


child_counter = 0
for child in resnet50.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

# plt.imshow(img); plt.show()
# to be changed to albumentations
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# removing avg. pool & FCN layers for both streams
image_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
click_block = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
image_block = image_block.cuda() if use_gpu else image_block
click_block = click_block.cuda() if use_gpu else click_block

for param in image_block.parameters():
    param.requires_grad = False

for param in click_block.parameters():
    param.requires_grad = False




#fcn.fc = nn.Identity()
#print("Updated ", fcn.fc)

#out = fcn(inp)['out']
#print(out.shape)

#om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
#print(om.shape)
#print(np.unique(om))


# def decode_segmap(image, nc=21):
#
#   label_colors = np.array([(0, 0, 0),  # 0=background
#                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
#                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
#                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
#                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
#                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
#
#   r = np.zeros_like(image).astype(np.uint8)
#   g = np.zeros_like(image).astype(np.uint8)
#   b = np.zeros_like(image).astype(np.uint8)
#
#   for l in range(0, nc):
#     idx = image == l
#     r[idx] = label_colors[l, 0]
#     g[idx] = label_colors[l, 1]
#     b[idx] = label_colors[l, 2]
#
#   rgb = np.stack([r, g, b], axis=2)
#   return rgb


#rgb = decode_segmap(om)
#plt.imshow(rgb); plt.show()
#print(decoder)
#print(decoder(x))





#x = torch.randn(64, 8, 3, 3)

# decoder = nn.Sequential(
#     nn.ConvTranspose2d(8, 64, kernel_size=3, stride=1),
#     nn.ReLU(),
#     nn.Conv2d(64, 32, kernel_size=4, stride=2),
#     nn.ReLU(),
#     nn.Conv2d(32, 512, kernel_size=8, stride=4),
#     nn.ReLU(),
#     nn.Sigmoid()
# )

# # needs to be tested inside a module def:
# for child in image_block.children():
#     c0 = child[0]
#     c1 = child[1]
#     c2 = child[2]
#     c3 = child[3]
#     print(child)
#     break
# tester = nn.Sequential(c0, c1, c2, c3)
# print(tester)
