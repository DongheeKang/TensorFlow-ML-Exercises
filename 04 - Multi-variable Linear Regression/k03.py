from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_data = [[1, 1, 0],
          [1, 0, 2],
          [1, 3, 0],
          [1, 0, 4],
          [1, 5, 0]]
y_data = [1, 2, 3, 4, 5]

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
