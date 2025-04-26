import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import gym
from gym import spaces

# === Descarga de recursos necesarios ===
nltk.download('stopwords')
nlp = spacy.load("es_core_news_sm")
stop_words = set(stopwords.words("spanish"))
stop_words.update(["producto", "cliente", "servicio"])  # Stopwords personalizadas

# === Función de preprocesamiento con spaCy ===
def preprocesar_texto(texto):
    doc = nlp(texto.lower())
    return [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

# === Dataset simulado de ejemplo ===
corpus = ["Me encantó el producto", "Fue terrible", "Muy bueno", "No me gustó"]
etiquetas = ["positivo", "negativo", "positivo", "negativo"]  # Etiquetas en texto
df = pd.DataFrame({'texto': corpus, 'etiqueta': etiquetas})

# === Preprocesamiento del texto para Word2Vec (opcional si no se usa para la SVM) ===
corpus_tokenizado = [preprocesar_texto(texto) for texto in df["texto"]]
modelo_w2v = Word2Vec(sentences=corpus_tokenizado, vector_size=100, window=5, min_count=1, workers=4)

# === Codificación de etiquetas a valores numéricos ===
encoder = LabelEncoder()
df["etiqueta_codificada"] = encoder.fit_transform(df["etiqueta"])

# === División del conjunto de datos ===
X_train, X_test, y_train, y_test = train_test_split(df["texto"], df["etiqueta_codificada"], test_size=0.5)

# === Vectorización con TF-IDF ===
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# === Entrenamiento del modelo SVM ===
modelo_svm = SVC()
modelo_svm.fit(X_train_tfidf, y_train)

# === Predicción y evaluación ===
preds = modelo_svm.predict(X_test_tfidf)
print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, preds, target_names=encoder.classes_))

# === Visualización: Matriz de confusión ===
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - SVM")
plt.tight_layout()
plt.show()

# === RL: Definición de entorno personalizado ===
class EntornoSimple(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)
        self.state = 0

    def step(self, action):
        reward = 1 if action == (self.state % 2) else -1
        self.state = (self.state + 1) % 5
        done = self.state == 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

# === RL: Q-Learning ===
env = EntornoSimple()
q_table = np.zeros((5, 2))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
recompensas = []

for ep in range(100):
    estado = env.reset()
    done = False
    total = 0
    while not done:
        if np.random.rand() < epsilon:
            accion = env.action_space.sample()
        else:
            accion = np.argmax(q_table[estado])

        nuevo_estado, recompensa, done, _ = env.step(accion)
        q_table[estado, accion] += alpha * (recompensa + gamma * np.max(q_table[nuevo_estado]) - q_table[estado, accion])
        total += recompensa
        estado = nuevo_estado
    recompensas.append(total)

# === Visualización: Recompensa acumulada por episodio ===
plt.figure(figsize=(8, 4))
plt.plot(recompensas, color='green')
plt.title("Recompensa acumulada por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Visualización: Mapa de calor de Q-values ===
plt.figure(figsize=(6, 4))
sns.heatmap(q_table, annot=True, cmap='YlGnBu')
plt.title("Mapa de calor de Q-Values")
plt.xlabel("Acciones")
plt.ylabel("Estados")
plt.tight_layout()
plt.show()

