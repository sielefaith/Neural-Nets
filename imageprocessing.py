import cv2
import nnfs
import numpy as np
import matplotlib.pyplot as plt

#image_data = cv2.imread('tshirt.png', cv2.IMREAD_UNCHANGED)

#plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB ))
#plt.show()

image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image_data, cmap='gray')
plt.show()

#image_data = cv2.resize(image_data,(28,28))
#plt.show()

