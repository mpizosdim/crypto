from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
from generalFunctions import preprocess_and_create_data
import matplotlib.pyplot as plt

def init_and_fit_model(train_X, train_Y, test_X, test_Y, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_X.shape[2]))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        validation_data=(test_X, test_Y))
    return model


def evaluate_model(model, scaler_model, test_X, test_y, plot=True):
    y_hat = model.predict(test_X)
    # invert scaling for forecast
    inv_yhat = scaler_model.inverse_transform(y_hat)
    inv_yreal = scaler_model.inverse_transform(test_y)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_yreal, inv_yhat))
    if plot:
        f, (axs) = plt.subplots(inv_yhat.shape[1], sharex=True)
        for i in range(inv_yhat.shape[1]):
            y_p = inv_yhat[:, i]
            y_r = inv_yreal[:, i]
            axs[i].plot(y_p, color='red')
            axs[i].plot(y_r, color='blue')

        plt.show()
    else:
        for i in range(inv_yhat.shape[1]):
            y_p = inv_yhat[:, i]
            y_r = inv_yreal[:, i]
            print("===" * 5)
            print("y predicted: %s" %str(y_p))
            print("y real: %s" % str(y_r))
    print('Test RMSE: %.3f' % rmse)

if __name__ == '__main__':
    epochs = 20
    batch_size = 72
    train_X, train_y, test_X, test_y, scaler_model = preprocess_and_create_data()
    model = init_and_fit_model(train_X, train_y, test_X, test_y, epochs, batch_size)
    evaluate_model(model, scaler_model, test_X, test_y, plot=True)







