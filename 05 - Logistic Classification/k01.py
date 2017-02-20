import numpy as np
import os
import gc
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# Prepare dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(script_dir + '/train.txt',
                        unpack=True, dtype='float32')

x_data = train_data[0:-1].transpose().astype('float32')
y_data = train_data[-1].astype('float32')

# create model for logistic regression
model = Sequential()
model.add(Dense(2, input_dim=len(x_data[0])))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

# prints summary of the model to the terminal
model.summary()

# feed the data to the model
# y has to be encoded in one hot vector
model.fit(x_data, to_categorical(y_data, nb_classes=2),
          nb_epoch=100, verbose=0)

# test the data
p1 = model.predict(np.asarray([[1, 2, 2]]))
print(np.argmax(p1, axis=1) == 1)
p2 = model.predict(np.asarray([[1, 5, 5]]))
print(np.argmax(p2, axis=1) == 1)
p3 = model.predict(np.asarray([[1, 4, 3], [1, 3, 5]]))
print(np.argmax(p3, axis=1) == 1)

# mitigate tensorflow error
gc.collect()
