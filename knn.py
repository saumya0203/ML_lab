import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def predict_single(self, x):
        distances = []
        for x_train in self.X_train:
            dist = self.euclidean_distance(x, x_train)
            distances.append(dist)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    def predict(self, X):
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
X = np.array([[1, 2],
              [2, 3],
              [3, 3],
              [6, 7],
              [7, 8]])

y = np.array([0, 0, 0, 1, 1])
model = KNN(k=3)
model.fit(X, y)
X_test = np.array([[5, 6], [2, 2]])
predictions = model.predict(X_test)
print("Predictions:", predictions)