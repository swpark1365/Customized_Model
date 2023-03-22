from Customized_Model.Neural_Network import SNN, DNN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def make_dataset(data, test_size, window_size, forecast_horizon):
    train_x, train_y, test_x, test_y = [], [], [], []
    for index in range(0, len(data) - window_size - (forecast_horizon - 1)):
        if index < len(data) - test_size - window_size - (forecast_horizon - 1):
            train_x.append(data[index:index + window_size])
            train_y.append(data[index + window_size :index + window_size + forecast_horizon])
        else:
            test_x.append(data[index:index + window_size])
            test_y.append(data[index + window_size:index + window_size + forecast_horizon])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
def start():
    data = pd.read_csv("./Dataset/GEFCOM2012/GEFCOM2012_Zone_1.csv")
    data = data.loc[(data["Year"]==2004),:]
    load_data = data["Load"].values
    scaler = MinMaxScaler()
    load_data = scaler.fit_transform(load_data.reshape(-1, 1))
    load_data = list(load_data.reshape(1, -1)[0])
    train_x, train_y, test_x, test_y = make_dataset(load_data, 24*31, 24*7, 24)

    model = SNN(
        input_dim = 24*7,
        output_dim = 24,
        activation="selu",
        solver="adam",
        batch_size=200,
        learning_rate=0.001,
        epoch=100
    )

    # model = DNN(
    #     input_dim=24 * 7,
    #     output_dim=24,
    #     num_hidden=5,
    #     activation="selu",
    #     solver="adam",
    #     batch_size=200,
    #     learning_rate=0.001,
    #     epoch=100
    # )

    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    print(prediction)