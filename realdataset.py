import numpy as np
import cv2
import os
import nnfs

nnfs.init()

#Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
	#scan all directories and create list of labels
	labels = os.listdir(os.path.join(path, dataset))
	#create lists for samples and labels
	X = []
	y = []
	#for each label folder
	for label in labels:
		#And for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):
			#read the image
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
			#And append it and a label to the lists
			X.append(image)
			y.append(label)
	#convert data to proper numpy arrays and return
	return np.array(X), np.array(y).astype('uint8')

#MNIST dataset (train + test)	
def create_data_mnist(path):
	#load both sets separately
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)
	#And return all the data
	return X, y, X_test, y_test
	
#create dataset	
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
#scalling
X =(X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
#print(X.min(), X.max())
#print(X.shape)
keys = np.array(range(X.shape[0]))
#print(keys[:10])
#shuffle these keys
np.random.shuffle(keys)
print(keys[:10])
#new order of indexes
X = X[keys]
y = y[keys]
print(y[:15])

import matplotlib.pyplot as plt
#reshape as image is a vector already
plt.imshow((X[4].reshape(28, 28)))
plt.show()
#check class at same index
print(y[4])