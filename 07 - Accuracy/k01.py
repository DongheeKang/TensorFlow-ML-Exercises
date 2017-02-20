import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# Prepare dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X original shape (60000, 28, 28)
# y original shape (60000,)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
# Training matrix shape (60000, 784)
# Testing matrix shape (10000, 784)

Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]

# create model for logistic regression
model = Sequential()
model.add(Dense(10, input_dim=len(X_train[0])))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1), metrics=['accuracy'])

# prints summary of the model to the terminal
model.summary()

# feed the data to the model
# y has to be encoded in one hot vector
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          nb_epoch=25, batch_size=100, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
