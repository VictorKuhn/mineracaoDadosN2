# Previsão de Séries Temporais para Dados Climáticos

### Alunos: Davi Prudente Ferreira, Victor Hugo Bosse Kuhn e Wesley Erik Sardi.

Este projeto utiliza duas abordagens diferentes para prever dados climáticos usando dados históricos de temperatura para a região de Nova York. As abordagens utilizadas são LSTM (Long Short-Term Memory) e ARIMA (AutoRegressive Integrated Moving Average).

## Abordagens

### 1. LSTM (Long Short-Term Memory)

#### Passos Realizados:
1. **Carregamento e Pré-processamento dos Dados:**
   - Carregamento dos dados de temperatura.
   - Conversão da coluna de datas para o formato `datetime`.
   - Remoção de duplicatas e definição da data como índice.
   - Preenchimento de valores ausentes e remoção de outliers.
   - Normalização dos dados usando `MinMaxScaler`.

2. **Criação do Conjunto de Dados para o Modelo:**
   - Divisão dos dados em conjuntos de treinamento e teste.
   - Criação de janelas de tempo para entrada no modelo LSTM.

3. **Construção e Treinamento do Modelo LSTM:**
   - Construção do modelo com duas camadas LSTM e camadas Dense.
   - Compilação do modelo usando `Adam` como otimizador e `mean_squared_error` como função de perda.
   - Treinamento do modelo com 50 épocas.

4. **Fazer Previsões:**
   - Previsões nos dados de treinamento e teste.
   - Cálculo do erro quadrático médio (MSE) para avaliar o desempenho.

5. **Realização de Previsões Futuras:**
   - Previsão dos próximos 30 dias com base nos dados de teste.

#### Resultados:
- **Erro Quadrático Médio:**
  - Treinamento: 8.6410
  - Teste: 8.9884
- **Gráficos:**
  - As previsões do modelo LSTM seguem de perto os dados históricos e as previsões futuras são coerentes com os padrões observados.

#### Vantagens:
- Capacidade de capturar dependências de longo prazo nas séries temporais.
- Eficaz para dados não lineares e complexos.

#### Desvantagens:
- Necessidade de grandes quantidades de dados e poder computacional para treinamento.
- Tempo de treinamento pode ser longo, especialmente com muitas épocas.

### 2. ARIMA (AutoRegressive Integrated Moving Average)

#### Passos Realizados:
1. **Carregamento e Pré-processamento dos Dados:**
   - Carregamento dos dados de temperatura.
   - Conversão da coluna de datas para o formato `datetime`.
   - Remoção de duplicatas e definição da data como índice.
   - Preenchimento de valores ausentes.

2. **Divisão dos Dados em Treinamento e Teste:**
   - Divisão dos dados em conjuntos de treinamento e teste.

3. **Construção e Treinamento do Modelo ARIMA:**
   - Construção do modelo ARIMA com ordem (1, 1, 1) e ordem sazonal (1, 1, 1, 12).
   - Treinamento do modelo nos dados de treinamento.

4. **Fazer Previsões:**
   - Previsões nos dados de teste.
   - Cálculo do erro quadrático médio (MSE) para avaliar o desempenho.

5. **Realização de Previsões Futuras:**
   - Previsão dos próximos 30 dias com base nos dados históricos.

#### Resultados:
- **Erro Quadrático Médio:**
  - Teste: 636.4958
- **Gráficos:**
  - As previsões do modelo ARIMA seguem de perto os dados históricos, mas têm um desempenho inferior ao modelo LSTM, como indicado pelo MSE mais alto.

#### Vantagens:
- Simplicidade e rapidez no treinamento.
- Eficaz para séries temporais univariadas com padrões lineares.

#### Desvantagens:
- Limitações em capturar padrões não lineares e complexos.
- Pode não ser eficaz para séries temporais com dependências de longo prazo.

## Conclusões

Comparando os resultados das duas abordagens, o modelo LSTM apresentou um desempenho significativamente melhor em termos de erro quadrático médio, tanto no conjunto de treinamento quanto no de teste. Isso se deve à capacidade do LSTM de capturar dependências de longo prazo e padrões não lineares nos dados.

## Instruções para Execução

### Dependências
- Python 3.x
- Bibliotecas: pandas, numpy, matplotlib, sklearn, tensorflow, statsmodels

### Execução
1. Clone o repositório:
git clone <URL do repositório>

2. Navegue até o diretório do projeto:
cd <nome do diretório>

3. Instale as dependências:
pip install pandas numpy matplotlib sklearn tensorflow statsmodels

4. Execute o script LSTM:
python lstm_algorithm.py

5. Execute o script ARIMA:
python arima_algorithm.py

## Explicação das Epochs no LSTM

No treinamento de redes neurais, uma "epoch" representa uma iteração completa sobre todo o conjunto de dados de treinamento. Durante cada epoch, o modelo processa todos os exemplos de treinamento uma vez e ajusta os pesos dos neurônios com base nos erros encontrados.

**Por que usar múltiplas epochs?**
- **Aprendizagem Gradual:** Permite que o modelo aprenda de forma gradual, ajustando os pesos de maneira iterativa para minimizar a função de perda.
- **Convergência:** Múltiplas epochs ajudam a garantir que o modelo converja para um mínimo local ou global na superfície da função de perda.
- **Generalização:** Treinar por múltiplas epochs ajuda a melhorar a capacidade do modelo de generalizar para novos dados.

**Tempo de Treinamento:**
- O tempo de treinamento pode ser significativo, especialmente com grandes conjuntos de dados e modelos complexos, como LSTM. Isso ocorre porque cada epoch envolve a passagem de todos os dados de treinamento pelo modelo e o ajuste dos pesos com base nos erros.

### Considerações Finais

Este projeto demonstrou a aplicação de duas técnicas populares de previsão de séries temporais: LSTM e ARIMA. Através da comparação dos resultados, ficou claro que o modelo LSTM, apesar de mais complexo e demorado para treinar, apresentou um desempenho superior em termos de precisão de previsão. No entanto, a escolha entre LSTM e ARIMA depende das características específicas dos dados e dos recursos computacionais disponíveis.

---

## Arquivos do Projeto

- `lstm_algorithm.py`: Implementação do modelo LSTM para previsão de temperatura.
- `arima_algorithm.py`: Implementação do modelo ARIMA para previsão de temperatura.
- `climate_data_ny.csv`: Dataset de dados históricos de temperatura para a região de Nova York.

## Descrição dos Campos do Dataset

O dataset utilizado para a previsão de dados climáticos contém os seguintes campos:

- **STATION**: Código da estação meteorológica que registrou os dados.
- **NAME**: Nome da estação meteorológica e sua localização.
- **DATE**: Data da observação.
- **DAPR**: Precipitação diária acumulada (em polegadas) para a precipitação observada e medição de ponto único.
- **DASF**: Profundidade da neve diária acumulada (em polegadas) para a precipitação observada e medição de ponto único.
- **MDPR**: Precipitação média diária (em polegadas) para o mês, com base em medições múltiplas durante o dia.
- **MDSF**: Profundidade média diária da neve (em polegadas) para o mês, com base em medições múltiplas durante o dia.
- **PRCP**: Precipitação diária (em polegadas).
- **SNOW**: Acumulação de neve diária (em polegadas).
- **SNWD**: Profundidade da neve no solo (em polegadas).
- **TAVG**: Temperatura média diária (em graus Fahrenheit).
- **TMAX**: Temperatura máxima diária (em graus Fahrenheit).
- **TMIN**: Temperatura mínima diária (em graus Fahrenheit).
- **TOBS**: Temperatura observada (em graus Fahrenheit) no horário de observação da estação.
- **WT01**: Condição de tempo 01 (por exemplo, nevoeiro, fumaça).
- **WT02**: Condição de tempo 02 (por exemplo, chuva leve ou chuvisco).
- **WT03**: Condição de tempo 03 (por exemplo, tempestade de neve).
- **WT04**: Condição de tempo 04 (por exemplo, chuva ou granizo).
- **WT05**: Condição de tempo 05 (por exemplo, nevoeiro congelante).
- **WT06**: Condição de tempo 06 (por exemplo, granizo).
- **WT08**: Condição de tempo 08 (por exemplo, tempestade de poeira ou areia).
- **WT09**: Condição de tempo 09 (por exemplo, nevasca).
- **WT11**: Condição de tempo 11 (por exemplo, tempestade de gelo).
