from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

tester = torch.tensor(np.ones([224, 224, 3]))
#tester = torch.unsqueeze(tester.T, 0)
print(tester.shape)
plt.ion()

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
print(vgg16)

print(vgg16.classifier[6].out_features)
# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

class_names = np.arange(10)
# Newly created modules have require_grad=True by default
#num_features = vgg16.classifier[6].in_features
#features = list(vgg16.classifier.children())[:-1] # Remove last layer
#features = list(vgg16.features)
#features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
#vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
#print(features)



vgg16.avgpool = Identity()
print(vgg16)
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])
print(vgg16)
vgg16.classifier = Identity()
print(vgg16)


out = vgg16(tester.T)
print(out.shape)
