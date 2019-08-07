# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:24:30 2019

@author: user
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist.load_data()

(train_imgaes, train_labels) , (test_images, test_labels) = fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_imgaes.shape
len(train_imgaes)
set(train_labels)

test_images.shape
len(test_images)

plt.figure()
plt.imshow(train_imgaes[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_imgaes = train_imgaes/255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgaes[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, tf.nn.softmax)
        ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_imgaes, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Acc: ", test_acc)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]


## graph images

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
               100 *np.max(predictions_array),
               class_names[true_label]), color = color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot= plt.bar(range(10), predictions_array, color= '#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


num_rows = 5
num_cols = 3

num_img = num_rows * num_cols 

plt.figure(figsize= (2 * 2 * num_cols, 2 * num_rows))

for i in range(num_img):
    plt.subplot(num_rows, 2 * num_cols, 2*i+1)
    plot_image(i,predictions,test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2*i +2)
    plot_value_array(i, predictions, test_labels)
plt.show()


# test model on test image

img = test_images[0]
img.shape

img = np.expand_dims(img, 0)
print(img.shape)

prediction_single = model.predict(img)

plot_value_array(0, prediction_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()