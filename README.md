# Gelato Mágico - Previsão de Vendas de Sorvete com Machine Learning

Este projeto utiliza Machine Learning para prever as vendas diárias de sorvete com base na temperatura ambiente, ajudando a sorveteria *Gelato Mágico* a planejar sua produção de forma eficiente.

## Objetivo
Desenvolver um modelo de regressão preditiva que:
- Treine um modelo para prever vendas com base na temperatura.
- Use MLflow para gerenciar experimentos.
- Implemente previsões em tempo real (simulado localmente).
- Crie um pipeline reprodutível.

## Estrutura do Repositório
- `inputs/`: Contém os dados iniciais (ex.: `dados_sorvete.txt`).
- `modelo.py`: Script Python com o pipeline de treinamento e previsão.
- `README.md`: Documentação do projeto.

## Processo
1. **Coleta de Dados**: Usei um arquivo de texto simples (`inputs/dados_sorvete.txt`) com temperaturas e vendas fictícias.
2. **Modelo**: Implementei uma regressão linear simples com a biblioteca Scikit-learn.
3. **MLflow**: Registrei métricas como MAE (Mean Absolute Error) e R².
4. **Previsão**: Criei uma função para prever vendas em tempo real com base em uma temperatura inserida.

## Código Exemplo
Aqui está um trecho do modelo implementado:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow

# Dados fictícios
data = {'Temperatura': [30, 25, 35, 20], 'Vendas': [150, 100, 200, 80]}
df = pd.DataFrame(data)

# Treinamento
X = df[['Temperatura']]
y = df['Vendas']
model = LinearRegression()
model.fit(X, y)

# Registro no MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("R2", model.score(X, y))