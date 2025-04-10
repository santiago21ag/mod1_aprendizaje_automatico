import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

# --- Cargar datasets ---
# Aquí cargamos los datos desde los archivos CSV. Asegúrate de que las rutas sean correctas.
diabetes_data = pd.read_csv('diabetes.csv')  # Datos para el modelo de regresión
cancer_data = pd.read_csv('breast_cancer.csv')  # Datos para el modelo de clasificación


# --- Preprocesamiento general ---
def preprocesamiento(df, target_column, drop_columns=None):
    """
    Preprocesa el DataFrame para preparar los datos para el entrenamiento:
    1. Elimina las columnas que no sean relevantes (si se indica).
    2. Rellena los valores NaN con la media de cada columna numérica.
    3. Codifica la columna objetivo si es categórica.
    4. Escala las características numéricas.

    Parametros:
    - df: DataFrame con los datos a procesar.
    - target_column: nombre de la columna objetivo (target).
    - drop_columns: columnas a eliminar antes de procesar (opcional).

    Retorna:
    - X_train, X_test: conjuntos de características de entrenamiento y prueba.
    - y_train, y_test: conjuntos de la variable objetivo de entrenamiento y prueba.
    """

    # Mostrar los valores nulos por columna antes de limpiar
    print("Valores nulos por columna antes de limpiar:\n", df.isnull().sum())

    if drop_columns:
        # Si hay columnas que no son relevantes, las eliminamos
        df = df.drop(columns=drop_columns)

    # Rellenamos los valores NaN en columnas numéricas con la media de cada columna
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Mostrar los valores nulos por columna después de limpiar
    print("Valores nulos por columna después de limpiar:\n", df.isnull().sum())

    # Codificar la columna objetivo (target) si es categórica
    if df[target_column].dtype == 'object':
        le = LabelEncoder()  # Crea un codificador para convertir texto en números
        df[target_column] = le.fit_transform(df[target_column])

    # Dividimos el DataFrame en características (X) y variable objetivo (y)
    X = df.drop(columns=[target_column])  # X son todas las columnas excepto la objetivo
    X = X.select_dtypes(include=[np.number])  # Aseguramos que solo estamos trabajando con columnas numéricas

    # Verificamos si hay NaN en las características antes del escalado
    print("¿Hay NaN en X antes de escalar?:", X.isnull().sum().sum())

    # Si hay valores NaN (por alguna razón), los rellenamos con 0
    X.fillna(0, inplace=True)  # O puedes usar dropna(axis=1) si prefieres eliminar columnas con NaN

    y = df[target_column]  # Variable objetivo

    # Escalamos las características para que todas estén en el mismo rango
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Ajustamos el escalador y transformamos X

    # Dividimos los datos en conjunto de entrenamiento y prueba (70% / 30%)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# --- Modelos de Regresión ---
def modelos_regresion(X_train, X_test, y_train, y_test):
    """
    Función que entrena varios modelos de regresión y muestra las métricas de evaluación.
    Modelos:
    - Regresión Lineal
    - Support Vector Regressor (SVR)
    - Random Forest Regressor

    Parametros:
    - X_train, X_test: Características de entrenamiento y prueba.
    - y_train, y_test: Etiquetas de entrenamiento y prueba.
    """

    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        # Entrenamos cada modelo con los datos de entrenamiento
        model.fit(X_train, y_train)

        # Realizamos predicciones con el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calculamos las métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio
        rmse = np.sqrt(mse)  # Raíz del error cuadrático medio
        r2 = r2_score(y_test, y_pred)  # R2, la métrica de ajuste del modelo

        # Imprimimos los resultados de cada modelo
        print(f'\n{name}')
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'R^2: {r2:.4f}')


# --- Modelos de Clasificación ---
def modelos_clasificacion(X_train, X_test, y_train, y_test):
    """
    Función que entrena varios modelos de clasificación y muestra las métricas de evaluación.
    Modelos:
    - Regresión Logística
    - Support Vector Classifier (SVC)
    - K-Nearest Neighbors

    Parametros:
    - X_train, X_test: Características de entrenamiento y prueba.
    - y_train, y_test: Etiquetas de entrenamiento y prueba.
    """

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Classifier': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    for name, model in models.items():
        # Entrenamos cada modelo con los datos de entrenamiento
        model.fit(X_train, y_train)

        # Realizamos predicciones con el conjunto de prueba
        y_pred = model.predict(X_test)

        # Imprimimos el reporte de clasificación (precisión, recall, f1-score, etc.)
        print(f'\n{name}')
        print(classification_report(y_test, y_pred))

        # Generamos la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.show()


# --- Ejecutar para Regresión (Diabetes) ---
# Ejecutamos el preprocesamiento para el conjunto de datos de diabetes y luego entrenamos los modelos de regresión
print("\n--- Resultados Regresión (Diabetes) ---")
X_train, X_test, y_train, y_test = preprocesamiento(diabetes_data, target_column='Outcome')
modelos_regresion(X_train, X_test, y_train, y_test)

# --- Ejecutar para Clasificación (Cáncer de Mama) ---
# Ejecutamos el preprocesamiento para el conjunto de datos de cáncer de mama y luego entrenamos los modelos de clasificación
print("\n--- Resultados Clasificación (Breast Cancer) ---")
X_train, X_test, y_train, y_test = preprocesamiento(cancer_data, target_column='diagnosis', drop_columns=['id'])
modelos_clasificacion(X_train, X_test, y_train, y_test)
