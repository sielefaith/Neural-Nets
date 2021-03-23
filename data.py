import matplotlib.pyplot as plt
import numpy as np
import cv2
image_data = cv2.imread('fashion_mnist_images/train/4/0011.png', cv2.IMREAD_UNCHANGED)
#print(image_data)
#np.set_printoptions(linewidth = 2000)

plt.imshow(image_data, cmap='gray')
plt.show()