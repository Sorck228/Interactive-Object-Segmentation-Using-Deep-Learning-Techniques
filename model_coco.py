from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision
from dataset import myOwnDataset, get_transform
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import sys


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# path to your own data and coco file
train_data_dir = '../../Github_data/coco/images/train2017'
train_json_dir = '../../Github_data/coco/annotations/instances_train2017.json'

#coco_train = dset.CocoDetection(root=train_data_dir, annFile=train_json_dir, transforms=get_transform())


# create own Dataset
my_dataset = myOwnDataset(root=train_data_dir,
                         annotation=train_json_dir,
                         transforms=get_transform()
                         )

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 100


# own DataLoader
data_loader = DataLoader(my_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=collate_fn)



# 2 classes; Only target class or background
num_classes = 2
num_epochs = 10
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

len_dataloader = len(data_loader)


# DataLoader is iterable over Dataset
# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     # print(annotations)

for epoch in range(num_epochs):
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        #print(imgs)
        #print(annotations)
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
