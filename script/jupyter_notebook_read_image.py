import cv2
import tensorflow as tf
from matplotlib import pyplot

# Read an external image
img = cv2.imread('/home/jovyan/work/fashion_example01.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Example01', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
pyplot.figure()
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pyplot.show()
print("Before preprossing, img.shape = ", img.shape)

# Resize and negate the image
imgsize = 28
img = cv2.bitwise_not(img)
img = cv2.resize(img, (imgsize, imgsize))
# cv2.imshow('Reduced01', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
pyplot.figure()
pyplot.imshow(img)
pyplot.show()

# Reshape the image for the CNN fashion model
img = tf.expand_dims(img, -1)
img = tf.divide(img, 255)
img = tf.reshape(img, [1, imgsize, imgsize, 1])
print("After preprossing, img.shape = ", img.shape)