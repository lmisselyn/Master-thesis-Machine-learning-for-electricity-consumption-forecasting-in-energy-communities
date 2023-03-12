import numpy as np
import pandas as pd
import tensorflow
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

import helper

# Load data from CSV file
variables = helper.get_features('../Datasets/one_year_10.csv')
df = pd.read_csv('../test.csv', index_col=0, parse_dates=["Datetime"])
x = np.transpose([df[var].to_numpy() for var in variables])
y = df["Consumption(Wh)"]

# Normalize data using MinMaxScaler
sc = StandardScaler()
scaler = sc.fit_transform(x)
print(x.shape)
# Create input and output sequences
input_seq = []
output_seq = []
window_size = 48
horizon_size = 12

for i in range(len(x) - window_size - horizon_size):
    input_seq.append(x[i:i + window_size])
    output_seq.append(y[i + window_size:i + window_size + horizon_size])

# Convert sequences to numpy arrays
input_seq = np.array(input_seq)
output_seq = np.array(output_seq)

# Split data into training and testing sets
train_size = int(len(input_seq) * 0.8)
test_size = len(input_seq) - train_size

train_input_seq = input_seq[:train_size]
train_output_seq = output_seq[:train_size]
test_input_seq = input_seq[train_size:]
test_output_seq = output_seq[train_size:]

# Build TDNN architecture
model = Sequential()
model.add(Dense(64, input_shape=(window_size,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(horizon_size, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train TDNN
model.fit(train_input_seq, train_output_seq, epochs=50, batch_size=64, verbose=2)

# Evaluate TDNN
test_pred_seq = model.predict(test_input_seq)
test_pred_seq = scaler.inverse_transform(test_pred_seq.reshape(-1, horizon_size))
test_output_seq = scaler.inverse_transform(test_output_seq.reshape(-1, horizon_size))

test_mae = np.mean(np.abs(test_pred_seq - test_output_seq))
test_rmse = np.sqrt(np.mean((test_pred_seq - test_output_seq)**2))
test_mape = np.mean(np.abs((test_pred_seq - test_output_seq) / test_output_seq)) * 100

print('Test MAE:', test_mae)
print('Test RMSE:', test_rmse)
print('Test MAPE:', test_mape)

# Make predictions for future time steps
future_input_seq = input_seq[-window_size:]

