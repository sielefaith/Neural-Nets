import cv2
import nnfs
import numpy as np
import matplotlib.pyplot as plt

#label index to label name relation
fashion_mnist_labels = {
0:'T-shirt / top',
1:'Trouser',
2:'Pullover',
3:'Dress',
4:'Coat',
5:'Sandal',
6:'Shirt',
7:'Sneaker',
8:'Bag',
9:'Ankle boot'
	}

image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)

image_data = cv2.resize(image_data,(28,28))
image_data = (image_data.reshape(1,-1).astype(np.float32)-127.5)/127.5
#load the model
model = Model.load('fashion_mnist.model')
#predict on the image
predictions = model.predict(image_data)
#get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(predictions)
#get label name from label index
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)

