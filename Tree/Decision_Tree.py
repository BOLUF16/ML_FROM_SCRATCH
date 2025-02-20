import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}

    def fit(self, X, y):
        """Builds the decision tree."""
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively grows the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        node = {}

        # Stop growing if max depth is reached or data is pure
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            node['type'] = 'leaf'
            node['value'] = Counter(y).most_common(1)[0][0]  # Most common class label
            return node

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)

        # If no split is found, return a leaf node
        if best_feature is None:
            node["type"] = 'leaf'
            node["value"] = Counter(y).most_common(1)[0][0]
            return node

        # Split data into left and right subsets
        left_split = X[:, best_feature] <= best_threshold
        right_split = X[:, best_feature] > best_threshold

        node["type"] = "branch"
        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left"] = self._grow_tree(X[left_split], y[left_split], depth + 1)
        node["right"] = self._grow_tree(X[right_split], y[right_split], depth + 1)

        return node

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split the data."""
        best_feature = None
        best_threshold = None
        min_gini = float('inf')

        for feature in range(X.shape[1]):
            thresholds = sorted(set(X[:, feature]))  # Get unique values

            for threshold in thresholds:
                weighted_gini = self._weighted_gini(y, X[:, feature], threshold)

                if weighted_gini < min_gini:
                    min_gini = weighted_gini
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _weighted_gini(self, y, X_column, threshold):
        """Computes the weighted Gini impurity for a split."""
        left_split = X_column < threshold
        right_split = X_column > threshold

        # If a split results in an empty group, return infinity (invalid split)
        if sum(left_split) == 0 or sum(right_split) == 0:
            return float('inf')

        n = len(y)
        n_l, n_r = sum(left_split), sum(right_split)
        g_l = self._gini_impurity(y[left_split])
        g_r = self._gini_impurity(y[right_split])

        # Compute weighted Gini impurity
        weighted_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return weighted_gini

    def _gini_impurity(self, y):
        """Calculates the Gini impurity of a dataset."""
        count = np.bincount(y)
        prob = count / len(y)
        return 1 - np.sum(prob ** 2)

    def predict(self, X):
        """Predicts class labels for input samples."""
        return [self._predict_sample(x) for x in X]

    def _predict_sample(self, x):
        """Predicts class label for a single sample."""
        node = self.tree

        # Traverse the tree until a leaf node is reached
        while node["type"] != "leaf":
            if x[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        return node["value"]

                







        


        


        


            

    


    
    


        