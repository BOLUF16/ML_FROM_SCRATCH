import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate = 0.01, alpha = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0

        for i in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            cost = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, cost) + self.alpha * np.square(self.weights)
            db = (1 / n_samples) * np.sum(cost)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias