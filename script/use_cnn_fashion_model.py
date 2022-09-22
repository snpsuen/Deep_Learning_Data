import tensorflow as tf
import cv2
from tensorflow import keras
from numpy import asarray, argmax

model = tf.keras.models.load_model("./cnn_fashion_class_model")
img = cv2.imread('./1042.png', cv2.IMREAD_GRAYSCALE)
print("Before preprossing, img.shape = " % (img.shape))

imgsize = 28
img = tf.expand_dims(img, -1)
img = tf.divide(img, 255)
img = tf.image.resize(img, [imgsize, imgsize])
img = tf.reshape(img, [1, imgsize, imgsize, 1])
print("After preprossing, img.shape = " % (img.shape))

classnames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
yhat = model.predict(img)
print('Predicted class for 1024.png = %s' % classnames[argmax(yhat)])
