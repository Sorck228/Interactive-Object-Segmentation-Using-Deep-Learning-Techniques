from utils import compare_mask, sim_first_click, make_test_images
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature

org_image, true_mask = make_test_images()
pred_mask = np.zeros((1280, 1918))
pred_mask[300:640, 500:1200] = 1
true_foreground_mask, false_background_mask, false_foreground_mask, true_background_mask = compare_mask(true_mask=true_mask, pred_mask=pred_mask)
fused_image, center = sim_first_click(true_mask=true_mask)
neg_dist_map = np.asarray(np.nonzero(feature.canny(false_foreground_mask, sigma=3)))




print(neg_dist_map[0][0])  # Height
print(neg_dist_map[1][0])  # Width
min_dist = np.inf
min_point = None

print(len(neg_dist_map[0]))
for i in range(len(neg_dist_map[0])):
    point = np.transpose(np.array([[neg_dist_map[1][i]], [neg_dist_map[0][i]]]))
    dist = np.linalg.norm(center - point)
    if dist < min_dist:
        min_dist = dist
        min_point = point[0]
        print('point ', point)
        print('distance ', dist)


print('min_dist ', min_dist)
print('min_point ', min_point)

pred_mask2 = np.zeros((1280, 1918))

pred_mask2[min_point[0], min_point[1]] = 1
false_foreground_mask = feature.canny(false_foreground_mask, sigma=1)
false_foreground_mask[center[1], center[0]] = 1
false_foreground_mask[min_point[1], min_point[0]] = 1
plt.figure("pred_mask2")
plt.imshow(false_foreground_mask, cmap='Greys_r')
plt.colorbar()
plt.show()


