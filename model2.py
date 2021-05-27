import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# setup
dataset = datasets.MNIST(
    root='/Users/albert/Documents/MNIST', download=True,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]))

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4)

device = 'cuda'
model = models.vgg16(pretrained=True)
model.features[0] = nn.Conv2d(1, 64, 3, 1, 1)
model.classifier[-1] = nn.Linear(4096, 1000)
model.classifier.add_module('7', nn.ReLU())
model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
model.classifier.add_module('9', nn.Linear(1000, 10))
model.classifier.add_module('10', nn.LogSoftmax(dim=1))

for param in model.features[1:].parameters():
    param.requires_grad = False

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

for epoch in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    for data, target in loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print('Epoch {}, loss {}, time {}'.format(
        epoch, loss.item(), (t1 - t0)))
