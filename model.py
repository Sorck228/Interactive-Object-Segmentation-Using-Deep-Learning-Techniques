from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
from albumentations.pytorch import ToTensorV2

fcn = models.resnet18(pretrained=True).eval()
img = Image.open("dog.jpeg")

plt.imshow(img); plt.show()

trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor()])
                 #T.Normalize(mean = [0.485, 0.456, 0.406],
                 #            std = [0.229, 0.224, 0.225])]

print(img.size)
inp = trf(img).unsqueeze(0)
print(inp.shape)

print(fcn)

print(fcn.fc)

fcn.fc = nn.Identity()
print("Updated ", fcn.fc)
# out = fcn(inp)['out']
# print(out.shape)
#
# om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
# print(om.shape)
# print(np.unique(om))

x = torch.randn(64, 8, 3, 3)

decoder = nn.Sequential(
    nn.ConvTranspose2d(8, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Conv2d(64, 32, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(32, 512, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Sigmoid()
)

print(decoder)
print(decoder(x))
