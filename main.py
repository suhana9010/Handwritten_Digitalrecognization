import tensorflow as tf

mnist = tf.keras.datasets.mnist

#Load the data set
(x_train,y_train),(x_test, y_test) = mnist.load_data()
x_train.shape

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

plt.imshow(x_train[0], cmap = plt.cm.binary)
print(x_train[0])

#Normalize the images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap=plt.cm.binary)

print(x_train[0])
print(y_train[0])

import numpy as np
IMG_SIZE=28
x_trainn = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testt = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples dimension", x_trainn.shape)
print("Testing Samples dimension", x_testt.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x_trainn.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()
print("Total training samples =",len(x_trainn))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_trainn,y_train, epochs=5, validation_split= 0.3)
test_loss, test_acc = model.evaluate(x_testt,y_test)

#check validation accuracy
print('Test loss =', test_loss)
print('valiation Accuracy =',test_acc)

predictions = model.predict([x_testt])
print(predictions)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])

print(np.argmax(predictions[128]))
plt.imshow(x_test[128])

#Testing it by my owm image
import cv2
img = cv2.imread('zero.png')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
resized = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_AREA)
resized.shape
#normalize the image
new = tf.keras.utils.normalize(resized, axis = 1)

new = np.array(new).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

new.shape

predictions = model.predict(new)
print(np.argmax(predictions))
