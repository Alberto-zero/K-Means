import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#Datos meh sacados de sklearn.datasets para la demostracion si se quiere demostrar muevan centers que pasa si no se 
# escoge bien la cantidad de centroides los datos son mas dispersos 
#donde n_samples es la cantidad de puntos, center es la cant de clusters, cluster_std la dispersion , random_state para reproducibilidad
#(como la semilla de un mundo de minecraft)
X, y_verdadero = make_blobs(n_samples=300, centers=6, cluster_std=1, random_state=7)



inercias = []
rango_k = range(1, 20)

for k in rango_k:
    modelo_k = KMeans(n_clusters=k, init='random', n_init=10, random_state=None)
    modelo_k.fit(X)
    inercias.append(modelo_k.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(rango_k, inercias, marker='o', color='b', linestyle='--')
plt.title('Método del Codo para Elegir K')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(rango_k)
plt.grid(True)
plt.show()

k_optimo = 6
plt.figure(figsize=(12, 4))
pasos_iteracion = [1, 2, 3] 

for i, iteracion in enumerate(pasos_iteracion):
    km_paso = KMeans(n_clusters=k_optimo, init='random', max_iter=iteracion, n_init=1, random_state=50)
    km_paso.fit(X)
    
    plt.subplot(1, 3, i + 1)
    # Graficar los puntos de los clientes con sus etiquetas temporales
    plt.scatter(X[:, 0], X[:, 1], c=km_paso.labels_, s=30, cmap='viridis', alpha=0.6)
    # Graficar los centroides en esa iteración específica
    centroides = km_paso.cluster_centers_
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=150, marker='X', edgecolors='black', label='Centroides')
    plt.title(f'Iteración {iteracion}')
    if i == 0:
        plt.legend()


modelo_final = KMeans(n_clusters=k_optimo, init='k-means++', n_init=10, random_state=42)
modelo_final.fit(X)


nuevos_clientes = np.array([
    [-2, 3],
    [4, 8],
    [2, 1]
])

predicciones = modelo_final.predict(nuevos_clientes)


print("--- PREDICCIONES DE NUEVOS CLIENTES ---")
for i, cliente in enumerate(nuevos_clientes):
    print(f"Nuevo Cliente {i+1} con características {cliente} -> Asignado al Segmento/Cluster: {predicciones[i]}")

plt.tight_layout()
plt.show()



