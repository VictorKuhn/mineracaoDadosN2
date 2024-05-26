import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Passo 1: Carregar os Dados
data = pd.read_csv('climate_data_ny.csv')

# Verificar os nomes das colunas
print(data.columns)

# Ajustar o nome da coluna de acordo com a saída acima
data['DATE'] = pd.to_datetime(data['DATE'])  # Substitua 'DATE' pelo nome correto da coluna se necessário

# Remover duplicatas
data = data.drop_duplicates(subset='DATE')

# Definir a coluna de data como índice
data.set_index('DATE', inplace=True)

# Verificar os dados após definir o índice
print(data.head())

# Verificar se a coluna de temperatura existe e está corretamente nomeada
if 'TAVG' in data.columns:
    temp_col = 'TAVG'
elif 'TMAX' in data.columns:
    temp_col = 'TMAX'
else:
    raise KeyError('Nenhuma coluna de temperatura encontrada no dataset')

# Passo 2: Exploração e Visualização dos Dados
plt.figure(figsize=(10, 6))
plt.plot(data[temp_col], label='Temperature')  # Usando a coluna correta de temperatura
plt.title('Historical Temperature Data - Arima')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Passo 3: Pré-processamento dos Dados
data = data.asfreq('D').fillna(method='ffill')

# Dividir os Dados em Treinamento e Teste
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Passo 4: Construir e Treinar o Modelo ARIMA
model = SARIMAX(train[temp_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Passo 5: Avaliar o Modelo
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
mse = mean_squared_error(test[temp_col], predictions)
print(f'Mean Squared Error: {mse}')

# Plotar as Previsões
plt.figure(figsize=(10, 6))
plt.plot(train[temp_col], label='Train')
plt.plot(test[temp_col], label='Test')
plt.plot(predictions, label='Predicted', color='red')
plt.title('Temperature Prediction - Arima')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Passo 6: Realizar Previsões Futuras
future_steps = 30  # Prever os próximos 30 dias
future_predictions = model_fit.predict(start=len(data), end=len(data) + future_steps - 1, dynamic=False)

plt.figure(figsize=(10, 6))
plt.plot(data[temp_col], label='Historical Data')
plt.plot(future_predictions, label='Future Predictions', color='red')
plt.title('Future Temperature Prediction - Arima')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
