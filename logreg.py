import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
class LogisticRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_pred_class)
X = np.array([[2, 3],
              [1, 1],
              [4, 5],
              [6, 7]])

y = np.array([0, 0, 1, 1])
model = LogisticRegression(lr=0.1, iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)