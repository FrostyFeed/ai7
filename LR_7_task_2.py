
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

# Завантажуємо набір даних Iris
iris = load_iris()
X = iris['data']
y = iris['target']

# Створюємо об'єкт KMeans з 3 кластерами (оскільки маємо 3 види ірисів)
# n_init=10 означає, що алгоритм буде запущено 10 разів з різними центроїдами
# random_state встановлюємо для відтворюваності результатів
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

# Навчаємо модель на наших даних
kmeans.fit(X)

# Отримуємо мітки кластерів для кожного зразка
y_kmeans = kmeans.predict(X)

# Візуалізуємо результати (використовуємо перші дві ознаки для 2D візуалізації)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Відображаємо центри кластерів
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3, label='Centroids')
plt.title('K-means clustering on Iris dataset')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()

# Створюємо власну функцію для реалізації K-means
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізуємо генератор випадкових чисел для відтворюваності
    rng = np.random.RandomState(rseed)
    
    # Випадково вибираємо початкові центри
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # Призначаємо точки до найближчого центру
        labels = pairwise_distances_argmin(X, centers)
        
        # Знаходимо нові центри як середнє значення точок у кожному кластері
        new_centers = np.array([X[labels == i].mean(0)
                              for i in range(n_clusters)])
        
        # Перевіряємо збіжність
        if np.all(centers == new_centers):
            break
            
        centers = new_centers
        
    return centers, labels

# Застосовуємо власну реалізацію
centers, labels = find_clusters(X, 3)

# Візуалізуємо результати власної реалізації
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Custom K-means implementation results')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()