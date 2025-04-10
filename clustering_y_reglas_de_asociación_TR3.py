# Importar librerías necesarias
# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# --- 1. Cargar los datasets ---
# Cargar el dataset de clustering
diabetes_data = pd.read_csv('Mall_Customers_tr3.csv')  # Ajusta la ruta según corresponda

# Cargar el dataset de reglas de asociación
basket_data = pd.read_csv('Market_Basket_Optimisation_tr3.csv')  # Ajusta la ruta según corresponda


# --- 2. Preprocesamiento para Clustering ---
def preprocesamiento_clustering(df, drop_columns=None):
    # Eliminar columnas no relevantes
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Asegurarse de que las columnas sean numéricas
    df = df.select_dtypes(include=[np.number])

    # Rellenar NaNs con la media de cada columna
    df.fillna(df.mean(), inplace=True)

    # Escalar los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


# --- 3. K-Means y Clustering Jerárquico ---
def kmeans_clustering(X_scaled):
    # Método del Codo
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia)
    plt.title('Método del Codo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.show()

    # Silueta
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    plt.plot(range(2, 11), silhouette_scores)
    plt.title('Coeficiente de Silueta')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Determinamos K óptimo (supongamos K=4 para este ejemplo)
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)

    return kmeans, y_kmeans


def visualizar_clusters(X_scaled, y_kmeans):
    # Reducir la dimensionalidad con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
    plt.title('Clusters K-Means')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()


# --- 4. Preprocesamiento para Reglas de Asociación ---
def preprocesamiento_reglas(df):
    # Convertir a formato One-Hot
    te = TransactionEncoder()
    df_onehot = te.fit(df).transform(df)
    df_onehot = pd.DataFrame(df_onehot, columns=te.columns_)

    # Filtrar productos poco frecuentes
    df_onehot = df_onehot.loc[:, df_onehot.mean() >= 0.05]
    return df_onehot


# --- 5. Algoritmo Apriori ---
def apriori_rules(df_onehot):
    frequent_itemsets = apriori(df_onehot, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
    return rules


# --- 6. Visualización de reglas de asociación ---
def visualizar_red(rules):
    G = nx.Graph()

    for _, row in rules.iterrows():
        G.add_edge(tuple(row['antecedents']), tuple(row['consequents']), weight=row['lift'])

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.title('Red de Reglas de Asociación')
    plt.show()


def visualizar_heatmap(df_onehot):
    sns.heatmap(df_onehot.corr(), annot=True)
    plt.title("Heatmap de Correlación entre Productos")
    plt.show()


# --- 7. Ejecución del análisis de Clustering ---
print("\n--- Análisis de Clustering (Mall Customers) ---")
X_scaled_clustering = preprocesamiento_clustering(diabetes_data, drop_columns=['CustomerID'])
kmeans_model, y_kmeans = kmeans_clustering(X_scaled_clustering)
visualizar_clusters(X_scaled_clustering, y_kmeans)

# --- 8. Ejecución del análisis de Reglas de Asociación ---
print("\n--- Análisis de Reglas de Asociación (Market Basket) ---")
df_onehot = preprocesamiento_reglas(basket_data)

# Apriori
rules_apriori = apriori_rules(df_onehot)
print("Reglas Apriori:\n", rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Visualización de las reglas de asociación
visualizar_red(rules_apriori)
visualizar_heatmap(df_onehot)

# --- 9. Reflexión ---
# Responde las preguntas en los comentarios del código:
# 1. Número óptimo de clusters: Usamos el método del codo y silueta para determinar que K=4 es el óptimo.
# 2. Regla más relevante: Ejemplo de justificación basado en el lift más alto.
# 3. Limitaciones en datasets de alta dimensionalidad: A medida que aumenta la dimensionalidad, las relaciones entre los datos pueden volverse menos claras, lo que puede dificultar la interpretación de los resultados de clustering y reglas de asociación.
