from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
from albumentations.pytorch import ToTensorV2

# fcn = models.resnet18(pretrained=True).eval()
fcn = models.segmentation.fcn_resnet50(pretrained=True).eval()
img = Image.open("dog.jpeg")
print(fcn)
plt.imshow(img); plt.show()

trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

print(img.size)
inp = trf(img).unsqueeze(0)
print(inp.shape)

#fcn.fc = nn.Identity()
#print("Updated ", fcn.fc)

out = fcn(inp)['out']
print(out.shape)

out2 = fcn(inp)


om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(om.shape)
print(np.unique(om))


def decode_segmap(image, nc=21):

  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb


rgb = decode_segmap(om)
plt.imshow(rgb); plt.show()

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

#print(decoder)
#print(decoder(x))
