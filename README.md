# Gelato M�gico - Previs�o de Vendas de Sorvete com Machine Learning

Este projeto utiliza Machine Learning para prever as vendas di�rias de sorvete com base na temperatura ambiente, ajudando a sorveteria *Gelato M�gico* a planejar sua produ��o de forma eficiente.

## Objetivo
Desenvolver um modelo de regress�o preditiva que:
- Treine um modelo para prever vendas com base na temperatura.
- Use MLflow para gerenciar experimentos.
- Implemente previs�es em tempo real (simulado localmente).
- Crie um pipeline reprodut�vel.

## Estrutura do Reposit�rio
- `inputs/`: Cont�m os dados iniciais (ex.: `dados_sorvete.txt`).
- `modelo.py`: Script Python com o pipeline de treinamento e previs�o.
- `README.md`: Documenta��o do projeto.

## Processo
1. **Coleta de Dados**: Usei um arquivo de texto simples (`inputs/dados_sorvete.txt`) com temperaturas e vendas fict�cias.
2. **Modelo**: Implementei uma regress�o linear simples com a biblioteca Scikit-learn.
3. **MLflow**: Registrei m�tricas como MAE (Mean Absolute Error) e R�.
4. **Previs�o**: Criei uma fun��o para prever vendas em tempo real com base em uma temperatura inserida.

## C�digo Exemplo
Aqui est� um trecho do modelo implementado:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow

# Dados fict�cios
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