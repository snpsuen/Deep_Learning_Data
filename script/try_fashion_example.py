import tensorflow as tf
import cv2
from tensorflow import keras
from numpy import asarray, argmax

img = cv2.imread('./fashion_example.jpg', cv2.IMREAD_GRAYSCALE)
print("Before preprossing, img.shape = " img.shape)

imgsize = 28
img = tf.expand_dims(img, -1)
img = tf.divide(img, 255)
img = tf.image.resize(img, [imgsize, imgsize])
img = tf.reshape(img, [1, imgsize, imgsize, 1])
print("After preprossing, img.shape = " img.shape)
