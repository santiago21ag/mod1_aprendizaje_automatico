import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym import spaces

# === Entorno personalizado ===
class EntornoSimple(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(5)  # 5 estados posibles
        self.action_space = spaces.Discrete(2)       # 2 acciones posibles (0 o 1)
        self.state = 0

    def step(self, action):
        # Recompensa positiva si acción coincide con paridad del estado
        reward = 1 if action == (self.state % 2) else -1
        self.state = (self.state + 1) % 5
        done = self.state == 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

# === Q-learning básico ===
env = EntornoSimple()
q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1       # Tasa de aprendizaje
gamma = 0.9       # Factor de descuento
epsilon = 0.2     # Probabilidad de exploración
num_episodios = 200

recompensas = []

for ep in range(num_episodios):
    estado = env.reset()
    done = False
    total = 0
    while not done:
        if np.random.rand() < epsilon:
            accion = env.action_space.sample()
        else:
            accion = np.argmax(q_table[estado])

        nuevo_estado, recompensa, done, _ = env.step(accion)
        # Actualización de la Q-table
        q_table[estado, accion] += alpha * (recompensa + gamma * np.max(q_table[nuevo_estado]) - q_table[estado, accion])
        total += recompensa
        estado = nuevo_estado

    recompensas.append(total)

# === Visualización: Recompensa acumulada por episodio ===
plt.plot(recompensas)
plt.title("Recompensa acumulada por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()

# === Visualización: Mapa de calor de la Q-table ===
plt.figure(figsize=(6, 4))
sns.heatmap(q_table, annot=True, cmap="YlGnBu")
plt.title("Mapa de calor de Q-values")
plt.xlabel("Acción")
plt.ylabel("Estado")
plt.show()
