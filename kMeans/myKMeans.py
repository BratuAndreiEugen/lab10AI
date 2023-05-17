import numpy as np


class MyKMeans:
    def __int__(self):
        self.__centroids = None

    def euclidean_distance(self, point1, point2):
        """Compute the Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def fit(self, data, k, max_iterations=100):
        """Perform k-means clustering on the given data."""
        num_samples, num_features = data.shape

        centroid_indices = np.random.choice(num_samples, size=k, replace=False)
        centroids = data[centroid_indices]

        for _ in range(max_iterations):
            labels = np.zeros(num_samples)
            for i in range(num_samples):
                distances = [self.euclidean_distance(data[i], centroid) for centroid in centroids]
                labels[i] = np.argmin(distances)

            new_centroids = np.zeros((k, num_features))
            counts = np.zeros(k)
            for i in range(num_samples):
                cluster_index = int(labels[i])
                new_centroids[cluster_index] += data[i]
                counts[cluster_index] += 1
            for j in range(k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            self.__centroids = new_centroids

        return labels, new_centroids

    def predict(self, test_data):
        """Assign cluster labels to test data based on the nearest centroids."""
        num_samples = test_data.shape[0]
        num_clusters = self.__centroids.shape[0]
        labels = np.zeros(num_samples)

        for i in range(num_samples):
            distances = [self.euclidean_distance(test_data[i], centroid) for centroid in self.__centroids]
            labels[i] = np.argmin(distances)

        return labels