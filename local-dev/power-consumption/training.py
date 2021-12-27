## Get the data
from azure.storage.blob import BlobServiceClient
import pandas as pd
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-k', '--key', type=str, required=True,
	help='azure connection string')
ap.add_argument('-f', '--file', type=str, default='household_power_consumption.txt',
	help='path to the processed dataset')
ap.add_argument('-b', '--blob', type=str, default='power-consumption/data',
	help='azure path for dataset blob')
ap.add_argument('-c', '--client', type=str, default='data',
	help='azure path for dataset blob')
ap.add_argument('-m', '--model', type=str, default='power_model.model',
	help='name for the trained model')
args = vars(ap.parse_args())

connect_str = args["key"]
azure_client = args["client"]

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(azure_client)

file_name = args["file"]
data_path = ".data"

if not os.path.exists(data_path):
    os.mkdir(data_path)
download_file_path = os.path.join(data_path, file_name)

azure_path = args["blob"]

blob_client = container_client.get_blob_client(os.path.join(azure_path, file_name))
with open(download_file_path, "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

df = pd.read_csv(download_file_path, sep=';', low_memory=False, 
    na_values=['nan','?'], parse_dates={'Date_time':['Date','Time']}, 
    index_col='Date_time', infer_datetime_format=True, header=0)

df['Sub_metering_4'] = (df.iloc[:,0] * 1000 / 60) - (df.iloc[:,4] + df.iloc[:,5] + df.iloc[:,6])
df.sort_values(by=["Date_time"], inplace=True)

## Save processed data
# processed_path = '.processed'
# if not os.path.exists(processed_path):
#     os.mkdir(processed_path)
# df.to_csv(os.path.join(processed_path, 'household-pow.csv'))

## Training

# Imports for model stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Deep-learing
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout

droping_list_all = []
for j in range(0, 8):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
droping_list_all

for j in range(0, 8):
    df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())
df.isnull().sum()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


## resampling of data over hour
df_resample = df.resample("h").mean()
df_resample.shape

# values = df.values
values = df_resample.values
# ensure all data is float
values = values.astype("float32")
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# split into train and test sets
values = reframed.values

n_train_time = 365 * 24
train = values[:n_train_time, :]
test = values[n_train_time:, :]
# test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
# model.add(LSTM(70), input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=20,
    batch_size=70,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False,
)

## Make a prediction
yhat = model.predict(test_X)
new_test_X = test_X.reshape((test_X.shape[0], 8))
inv_yhat = np.concatenate((yhat, new_test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
new_test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((new_test_y, new_test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)

model.save(args["model"], save_format="h5")
