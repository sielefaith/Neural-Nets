import os

labels = os.listdir('fashion_mnist_images/train')
print(labels)

files = os.listdir('fashion_mnist_images/train/0')
print(files[:10])
print(len(files))