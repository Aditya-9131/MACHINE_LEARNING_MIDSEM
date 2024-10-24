import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder  

df = pd.read_csv('Iris.csv')

data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
labels = df['Species'].values

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(data, k):
    centroids_idx = np.random.choice(data.shape[0], k, replace=False)
    return data[centroids_idx]

def assign_clusters(data, centroids):
    clusters = {}
    for idx, point in enumerate(data):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        nearest_centroid = np.argmin(distances)
        if nearest_centroid in clusters:
            clusters[nearest_centroid].append(idx)
        else:
            clusters[nearest_centroid] = [idx]
    return clusters

def update_centroids(data, clusters):
    centroids = np.zeros((len(clusters), data.shape[1]))
    for cluster_idx, points in clusters.items():
        centroids[cluster_idx] = np.mean(data[points], axis=0)
    return centroids

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for i in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def calculate_sse(data, centroids, clusters):
    sse = 0
    for centroid_idx, points in clusters.items():
        for idx in points:
            sse += euclidean_distance(data[idx], centroids[centroid_idx]) ** 2
    return sse

sse_values = []
for k in range(1, 11): 
    centroids, clusters = k_means(data, k)
    sse = calculate_sse(data, centroids, clusters)
    sse_values.append(sse)

plt.plot(range(1, 11), sse_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.show()

optimal_k = 3  

centroids, clusters = k_means(data, optimal_k)

def select_closest_samples(data, clusters, centroids, n=25):
    selected_indices = []
    for cluster_idx, points in clusters.items():
        distances = np.array([euclidean_distance(data[point], centroids[cluster_idx]) for point in points])
        sorted_idx = np.argsort(distances)[:n] 
        selected_indices.extend(np.array(points)[sorted_idx])
    return selected_indices

selected_indices = select_closest_samples(data, clusters, centroids)

train_data = data[selected_indices]
train_labels = encoded_labels[selected_indices]
test_indices = list(set(range(len(data))) - set(selected_indices))
test_data = data[test_indices]
test_labels = encoded_labels[test_indices]

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(train_data, train_labels)

train_pred = log_reg.predict(train_data)
test_pred = log_reg.predict(test_data)
train_acc = accuracy_score(train_labels, train_pred)
test_acc = accuracy_score(test_labels, test_pred)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
