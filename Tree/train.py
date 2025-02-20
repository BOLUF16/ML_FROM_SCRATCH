from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Tree.Decision_Tree import DecisionTree


data = datasets.load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

decisontree = DecisionTree(max_depth=2)
decisontree.fit(X_train, y_train)
predictions = decisontree.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

