import os
import random
import shutil
import pathlib
import math

# program to randomly sample images from directive into a test folder
# moves the file from training folder into test image and mask folder

# create empty file list
files_list = []

# image directive
destPath = "datasets/test"
# make directive if not existing
pathlib.Path(destPath).mkdir(parents=True, exist_ok=True)
# mask directive
mask_destPath = "datasets/test_masks"
# make directive if not existing
pathlib.Path(mask_destPath).mkdir(parents=True, exist_ok=True)

# get list of files in directive
for root, _, files in os.walk("datasets/train"):
    for file in files:
        # find all images and append to image file list
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            files_list.append(os.path.join(root, file))


test_ratio = 0.1
filesToCopy = random.sample(files_list, math.floor(len(files_list)*test_ratio))

# move image files
for file in filesToCopy:
    # print(file)
    shutil.move(file, destPath)
    # generate name of mask file to be moved
    if file.endswith(".jpg"):
        file = file.replace(".jpg", "_mask.gif")
    elif file.endswith(".png"):
        file = file.replace(".png", "_mask.gif")
    elif file.endswith(".jpeg"):
        file = file.replace(".jpeg", "_mask.gif")

    file = file.replace("datasets/train", "datasets/train_masks")  # ugly solution but works
    shutil.move(file, mask_destPath)

num_files = len(files_list)
num_train_files = len(files_list)-len(filesToCopy)
num_test_files = math.floor(len(files_list)*test_ratio)
print(f'No. of files {num_files} No. of train_files {num_train_files} No. of test_files moved {num_test_files}')


