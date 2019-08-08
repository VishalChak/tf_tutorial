# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:01:36 2019

@author: user
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data

print("Traning entries: {}, labels :{}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# convert integer back to words