import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(script_dir + '/train.txt', unpack=True, dtype='float32')

x_data = train_data[0:-1].transpose()
y_data = train_data[-1]

print(x_data)
print(x_data.shape)

# create model for linear regression
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=SGD(lr=0.1))

# prints summary of the model to the terminal
model.summary()

# feed the data to the model
model.fit(x_data, y_data, verbose=1)

# test the data
y_predict = model.predict(x_data)
print(y_predict)
