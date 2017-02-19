from keras.models import Model
from keras.layers import Dense, Input

x_data = [1, 2, 3]
y_data = [1, 2, 3]

x = Input((1,))
y = Dense(output_dim=1, input_dim=1, activation ='linear')(x)

model = Model(x,y)
model.compile(loss = 'mse', optimizer='sgd')
model.fit(x_data, x_data)

y_predict = model.predict(x_data)
print(y_predict)
