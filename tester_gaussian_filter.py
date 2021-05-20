from utils import normalize_data, make_click_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

org_image = np.array(Image.open('tester.jpg').convert("RGB"), dtype=np.float32)
org_image = normalize_data(org_image)
image_dim = np.array([org_image.shape[0], org_image.shape[1]])

clicks_pos2 = np.array([[959, 640], [100, 100], [500, 500], [1800, 1100]])  # [width, heigth]
clicks_pos = np.array([959, 640])
#clicks_neg = np.array([[1700, 700], [200, 800], [700, 700], [1000, 1000]])  # [width, heigth]
clicks_neg = np.array([959, 640])
#for click in clicks_pos:

print("clicks_pos2 shape")
print(clicks_pos2.shape)
print("loop start")
for click in clicks_pos2:
    print(click)
    print(click.shape)



gaussian_filter_pos = make_click_image(image_dim=image_dim, clicks=clicks_pos, first_click=True)
gaussian_filter_neg = make_click_image(image_dim=image_dim, clicks=clicks_neg)

fused_image = np.ones((int(org_image.shape[0]), int(org_image.shape[1]), 5), dtype=np.float32)

fused_image[:, :, :3] = org_image
fused_image[:, :, 3] = gaussian_filter_pos
fused_image[:, :, 4] = gaussian_filter_neg

print(gaussian_filter_pos.shape)
print(fused_image.shape)
# show image flags
show_gaussian = True
if show_gaussian:
    plt.figure("gaussian 2d pos")
    plt.imshow(gaussian_filter_pos)
    plt.colorbar()
    plt.show()
if show_gaussian:
    plt.figure("gaussian 2d neg")
    plt.imshow(gaussian_filter_neg)
    plt.colorbar()
    plt.show()
show_fused_image = False
if show_fused_image:
    plt.figure("fused image")
    plt.imshow(fused_image[:, :, :3])
    plt.show()
show_org_image = False
if show_org_image:
    plt.figure("image")
    plt.imshow(org_image)
    plt.show()


