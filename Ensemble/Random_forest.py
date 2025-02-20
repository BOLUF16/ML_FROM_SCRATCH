import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.selected_features = None  # Store selected features

    def _gini_impurity(self, y):
        """Calculates the Gini impurity of a dataset."""
        count = np.bincount(y)
        prob = count / len(y)
        return 1 - np.sum(prob ** 2)

    def _weighted_gini(self, y, X_column, threshold):
        """Computes the weighted Gini impurity for a split."""
        left_split = X_column <= threshold
        right_split = X_column > threshold

        # If a split results in an empty group, return infinity (invalid split)
        if sum(left_split) == 0 or sum(right_split) == 0:
            return float("inf")

        n = len(y)
        n_l, n_r = sum(left_split), sum(right_split)
        g_l = self._gini_impurity(y[left_split])
        g_r = self._gini_impurity(y[right_split])

        # Compute weighted Gini impurity
        weighted_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return weighted_gini

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split the data."""
        best_feature = None
        best_threshold = None
        min_gini = float("inf")

        # Use only the randomly selected features for this tree
        for feature in self.selected_features:
            thresholds = np.unique(X[:, feature])  # Get unique values

            for threshold in thresholds:
                weighted_gini = self._weighted_gini(y, X[:, feature], threshold)

                if weighted_gini < min_gini:
                    min_gini = weighted_gini
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """Recursively grows the decision tree."""
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        node = {}

        # Stop growing if max depth is reached or data is pure
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            node["type"] = "leaf"
            node["value"] = Counter(y).most_common(1)[0][0]  # Most common class label
            return node

        # Find the best split among sampled features
        best_feature, best_threshold = self._best_split(X, y)

        # If no split is found, return a leaf node
        if best_feature is None:
            node["type"] = "leaf"
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

    def fit(self, X, y, selected_features):
        """Builds the decision tree with only sampled features."""
        self.selected_features = selected_features  # Store sampled features
        self.tree = self._grow_tree(X, y)

    def _predict_sample(self, x):
        """Predicts class label for a single sample."""
        node = self.tree

        # Traverse the tree until a leaf node is reached
        while node["type"] != "leaf":
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        return node["value"]

    def predict(self, X):
        """Predicts class labels for input samples."""
        return np.array([self._predict_sample(x) for x in X])


class Randomforest:
    def __init__(self, max_depth = None, min_samples_split = None, n_trees = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            
            num_features = int(np.sqrt(n_features))
            slice = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[slice], y[slice]
            
            num_selected_features = min(n_features, num_features)
            selected_features = np.random.choice(n_features, num_selected_features, replace=False)

            tree.fit(X_sample, y_sample, selected_features)
            self.trees.append((tree, selected_features))

    def predict(self, X):
       tree_preds = np.array([tree.predict(X[:, selected_features]) for tree, selected_features in self.trees])

       return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(len(X))])


        