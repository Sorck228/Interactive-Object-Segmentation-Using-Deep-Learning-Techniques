from utils import normalize_data, compare_mask, make_test_images
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

org_image, true_mask = make_test_images()

pred_mask = np.zeros((1280, 1918))
pred_mask[300:640, 500:1200] = 1
true_foreground_mask, false_background_mask, false_foreground_mask, true_background_mask = compare_mask(true_mask=true_mask, pred_mask=pred_mask)


# plt.figure("true_mask")
# plt.imshow(true_mask, cmap='Greys_r')
# plt.colorbar()
# plt.show()
#
# plt.figure("test_mask")
# plt.imshow(test_mask, cmap='Greys_r')
# plt.colorbar()
# plt.show()
#
# plt.figure("correct mask part")
# plt.imshow(correct_mask, cmap='Greys_r')
# plt.colorbar()
# plt.show()
#
# plt.figure("missing mask part")
# plt.imshow(missing_mask, cmap='Greys_r')
# plt.colorbar()
# plt.show()
#
# plt.figure("wrong_mask")
# plt.imshow(wrong_mask, cmap='Greys_r')
# plt.colorbar()
# plt.show()


fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(true_mask, cmap='Greys_r')
axs[0, 0].set_title('Original mask')
axs[0, 1].imshow(true_foreground_mask, cmap='Greys_r')
axs[0, 1].set_title('True foreground pixels')
axs[0, 2].imshow(true_background_mask, cmap='Greys_r')
axs[0, 2].set_title('True background pixels')
axs[1, 0].imshow(pred_mask, cmap='Greys_r')
axs[1, 0].set_title('Predicted mask')
axs[1, 1].imshow(false_foreground_mask, cmap='Greys_r')
axs[1, 1].set_title('False foreground pixels')
axs[1, 2].imshow(false_background_mask, cmap='Greys_r')
axs[1, 2].set_title('False background pixels')

for ax in axs.flat:
    ax.set(xlabel='', ylabel='')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

