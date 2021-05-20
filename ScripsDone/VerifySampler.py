import os

train_list = []
# train_mask_list = []

test_list = []
# test_mask_list = []

for root, dirs, files in os.walk("datasets/train"):
    for file in files:
        # all
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            train_list.append(file)
for root, dirs, files in os.walk("datasets/test"):
    for file in files:
        # all
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            test_list.append(file)


def check_duplicates(test_list, train_list):
    result = False
    # traverse in the 1st list
    for x in test_list:
        # traverse in the 2nd list
        for y in train_list:
            # if one common
            if x == y:
                result = True
    return result


if check_duplicates(test_list, train_list):
    print("Duplicates found in train/test")
