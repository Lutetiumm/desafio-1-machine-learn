# Importar bibliotecas
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# irei explicar o oque foi feito nesse desafio sobre aprendizado supervisionado

# Treinei um modelo de machine learning para classificar tipos de vinho e ultilizei
# o dataset wine, que contém 13 features químicas e 3 classes de vinho.


# 1. Carregar o dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)  # Features
y = wine.target  # Target (classes de vinho)

# 2. Análise exploratória dos dados
print("=== Informações do Dataset ===")
print(f"Número de amostras: {X.shape[0]}")
print(f"Número de features: {X.shape[1]}")
print(f"Classes disponíveis: {wine.target_names}")

# Visualizar as primeiras linhas do dataset
print("\n=== Primeiras Linhas do Dataset ===")
print(X.head())

# Verificar a distribuição das classes
print("\n=== Distribuição das Classes ===")
distribuicao_classes = pd.Series(y).value_counts()
print(distribuicao_classes)

# Plotar a distribuição das classes
plt.figure(figsize=(8, 5))
sns.barplot(x=distribuicao_classes.index, y=distribuicao_classes.values, palette="viridis")
plt.title("Distribuição das Classes de Vinho")
plt.xlabel("Classe")
plt.ylabel("Número de Amostras")
plt.xticks(ticks=[0, 1, 2], labels=wine.target_names)
plt.show()

# Plotar a correlação entre as features
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação entre as Features")
plt.show()

# 3. Pré-processamento dos dados
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Treinamento do modelo
# Usar Random Forest para classificação
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Avaliação do modelo
# Fazer previsões
y_pred = model.predict(X_test)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print("\n=== Desempenho do Modelo ===")
print(f"Acurácia do modelo: {accuracy:.2f}")

# Plotar a acurácia do modelo
plt.figure(figsize=(6, 4))
sns.barplot(x=["Acurácia"], y=[accuracy], palette="Blues")
plt.ylim(0, 1.0)  # Limitar o eixo y entre 0 e 1
plt.title("Acurácia do Modelo")
plt.ylabel("Acurácia")
plt.show()

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n=== Matriz de Confusão ===")
print(conf_matrix)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Matriz de Confusão")
plt.xlabel("Previsão")
plt.ylabel("Verdadeiro")
plt.show()

# Relatório de classificação
print("\n=== Relatório de Classificação ===")
report = classification_report(y_test, y_pred, target_names=wine.target_names, output_dict=True)
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Plotar o relatório de classificação como uma tabela
plt.figure(figsize=(10, 4))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues", fmt=".2f")
plt.title("Relatório de Classificação")
plt.show()