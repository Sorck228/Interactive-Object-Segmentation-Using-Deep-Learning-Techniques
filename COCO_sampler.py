import os
import random
import shutil
import pathlib
import math

# program to randomly sample images from directive into a test folder
# moves the file from training folder into test image and mask folder

# create empty file list
files_list_img = []
files_list_mask = []
images_to_copy = []
masks_to_copy = []

train_data_dir = '../../Github_data/coco/images/train2017'
train_mask_dir = '../../Github_data/coco/annotations/stuff_train2017_pixelmaps'
img_dest_path = "../../Github_data/coco/img_reduced_sample"
mask_dest_path = "../../Github_data/coco/mask_reduced_sample"

# make directive if not existing
pathlib.Path(img_dest_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(mask_dest_path).mkdir(parents=True, exist_ok=True)

# get list of files in directive
for _, __, files in os.walk(train_data_dir):
    for file in files:
        # find all images and append to image file list
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            files_list_img.append(os.path.join(train_data_dir, file))
            files_list_mask.append(os.path.join(train_mask_dir, file).replace(".jpg", ".png"))

test_ratio = 0.1
# number_of_samples = int(len(files_list_img) * test_ratio)
# x = random.sample(range(0, len(files_list_img)), number_of_samples)
number_of_samples = 200
x = random.sample(range(0, len(files_list_img)), number_of_samples)

for index in range(len(x)):
    images_to_copy.append(files_list_img[x[index]])
    masks_to_copy.append(files_list_mask[x[index]])
    #print(images_to_copy[index], masks_to_copy[index])

# move files
for file in images_to_copy:
    shutil.copy(file, img_dest_path)
for file in masks_to_copy:
    shutil.copy(file, mask_dest_path)

num_files = len(files_list_img)
num_train_files = len(files_list_img) - len(images_to_copy)
num_test_files = math.floor(len(files_list_img) * test_ratio)

print(f'No. of files {num_files} No. of train_files {num_train_files} No. of test_files copied {num_test_files}')


