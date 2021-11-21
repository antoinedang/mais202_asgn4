import os, sys
from IPython.display import display, Image
import numpy as np

data = np.random.normal(0, 1, 100)
test_images_raw = np.load('test_images.npy')
training_images_raw = np.load('train_images.npy')

train_labels = open('train_labels.csv').readlines()[1:]


labels = { 0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot" }

for image in training_images_raw[:10]:
    display(PIL.Image.fromarray(image))