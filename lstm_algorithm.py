import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Passo 1: Carregar os Dados
data = pd.read_csv('climate_data_ny.csv')  # Certifique-se de especificar o caminho correto do arquivo CSV

# Verificar os nomes das colunas
print(data.columns)

# Ajustar o nome da coluna de acordo com a saída acima
data['DATE'] = pd.to_datetime(data['DATE'])

# Remover duplicatas
data = data.drop_duplicates(subset='DATE')

# Definir a coluna de data como índice
data.set_index('DATE', inplace=True)

# Verificar se a coluna de temperatura existe e está corretamente nomeada
if 'TAVG' in data.columns:
    temp_col = 'TAVG'
elif 'TMAX' in data.columns:
    temp_col = 'TMAX'
else:
    raise KeyError('Nenhuma coluna de temperatura encontrada no dataset')

# Preencher valores ausentes
data[temp_col] = data[temp_col].interpolate(method='time')

# Remover outliers
data = data[(np.abs(data[temp_col] - data[temp_col].mean()) <= (3 * data[temp_col].std()))]

# Reindexar os dados para frequência diária
data = data.asfreq('D').ffill()

# Passo 2: Pré-processamento dos Dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[temp_col].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Passo 3: Construir o Modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o Modelo com mais épocas
model.fit(X_train, y_train, batch_size=1, epochs=50)

# Passo 4: Fazer Previsões
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reverter a normalização
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular Erro Quadrático Médio
train_score = mean_squared_error(y_train, train_predict)
test_score = mean_squared_error(y_test, test_predict)
print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')

# Plotar as Previsões
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_predict_plot, label='Train Predict')
plt.plot(test_predict_plot, label='Test Predict')
plt.title('Temperature Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Passo 5: Realizar Previsões Futuras
x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = time_step
i = 0
while (i < 30):
    if (len(temp_input) > time_step):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

future_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Gerar datas futuras a partir da última data nos dados históricos
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=30, freq='D').tolist()

# Plotar as previsões futuras
plt.figure(figsize=(10, 6))
plt.plot(data.index, data[temp_col], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
plt.title('Future Temperature Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
