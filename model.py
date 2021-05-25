from torchvision import models
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np


fcn = models.resnet34(pretrained=True).eval()
img = Image.open("bird.png")

plt.imshow(img); plt.show()

trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor()])
                 #T.Normalize(mean = [0.485, 0.456, 0.406],
                 #            std = [0.229, 0.224, 0.225])]

print(img.size)
inp = trf(img).unsqueeze(0)
print(inp.shape)
out = fcn(inp)['out']
print(out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(om.shape)
print(np.unique(om))
