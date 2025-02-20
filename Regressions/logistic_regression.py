import numpy as np


class LogisticRegression:
    def __init__(self, epochs=1000, learning_rate=0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight = None
        self.bias = None

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, X):
        _, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        self.initialize_parameters(X)
        n_samples = X.shape[0]

        for i in range(self.epochs):
            # Compute linear function
            z = np.dot(X, self.weight) + self.bias
            A = self.sigmoid_function(z)

            # Compute cost function
            cost = -(1 / n_samples) * np.sum(y * np.log(A + 1e-9) + (1 - y) * np.log(1 - A + 1e-9))  # Avoid log(0) error

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (A - y))
            db = (1 / n_samples) * np.sum(A - y)

            # Update parameters using gradient descent
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print cost every 100 epochs
            if i % 100 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        A = self.sigmoid_function(z)
        return (A >= 0.5).astype(int)  # Convert probabilities to binary 0/1






    
