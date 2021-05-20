from utils import compare_mask, sim_first_click, sim_pos_click, sim_neg_click, make_dist, make_test_images
import numpy as np
from matplotlib import pyplot as plt

org_image, true_mask = make_test_images()

fused_image = np.ones((int(org_image.shape[0]), int(org_image.shape[1]), 5), dtype=np.float32)

fused_image[:, :, :3] = org_image
fused_image[:, :, 3], center = sim_first_click(true_mask=true_mask)

plt.figure("true mask")
plt.imshow(true_mask, cmap='Greys_r')
plt.colorbar()
plt.show()


pred_mask = np.zeros((1280, 1918))
pred_mask[300:640, 500:1200] = 1


true_foreground_mask, false_background_mask, false_foreground_mask, true_background_mask = compare_mask(true_mask=true_mask, pred_mask=pred_mask)
fused_image[:, :, 3] = np.add(sim_pos_click(false_background_mask=false_background_mask), fused_image[:, :, 3])
fused_image[:, :, 4] = np.add(sim_neg_click(false_foreground_mask=false_foreground_mask, center=center), fused_image[:, :, 4])


plt.figure("fused 3")
plt.imshow(fused_image[:, :, 3])
plt.colorbar()
plt.show()

plt.figure("fused 4")
plt.imshow(fused_image[:, :, 4])
plt.colorbar()
plt.show()

plt.figure("mask")
plt.imshow(make_dist(false_background_mask), cmap='Greys_r')
plt.colorbar()
plt.show()
# To generate positive clicks: find the center of missing mask and click
# To generate negative clicks: calculate a distance transform all wrongly labelled pixels
# to the center of the the correct mask then click the the areas with the lowest distance
# hence getting closest to the correct object

