import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0

        for i in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            cost = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, cost)
            db = (1 / n_samples) * np.sum(cost)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

    def predict(self, X, y):
        y_pred = np.dot(X, self.weights) + self.bias




            

