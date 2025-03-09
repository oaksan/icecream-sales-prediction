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

# Previsão em tempo real
temp = float(input("Digite a temperatura (°C): "))
predicao = model.predict([[temp]])
print(f"Previsão de vendas: {int(predicao[0])} sorvetes")
