from utils import make_dist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image = np.array(Image.open('../tester_mask.gif').convert("L"), dtype=np.float32)
image[image == 255.0] = 1.0
image[image <= 0.0] = 0.0

print(image.dtype)
print(image.shape)
# display the array of pixels as an image
plt.figure("distance map")
plt.imshow(image, cmap='Greys_r')
plt.colorbar()
plt.show()

dist = make_dist(image)
print(dist.dtype)
print(dist.shape)

plt.figure("distance map")
plt.imshow(dist, cmap='Greys_r')
plt.colorbar()
plt.show()

print(image.shape[0])
