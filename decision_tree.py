import pandas as pd
from sklearn.model_selection import train_test_split
from math import log
import random

Name = "HamidReza Yaghoubi"
Student_Number = "98109786"


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        if self.value is not None:
            return True
        return False


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def is_splitting_finished(self, depth, num_class_labels, num_samples):
        if depth == self.max_depth:
            return True

        if num_samples <= self.min_samples_split:
            return True

        if num_class_labels == 1:
            return True

        return False

    def split(self, X, y, feature, threshold):
        left_indexes = X[feature] <= threshold
        right_indexes = -left_indexes
        X_left = X[left_indexes]
        y_left = y[left_indexes]
        X_right = X[right_indexes]
        y_right = y[right_indexes]

        return X_left, X_right, y_left, y_right

    def entropy(self, y):
        p = len(y[y.target == 1]) / len(y)
        if p == 1 or p == 0:
            return 0
        return -p * log(p, 2) - (1 - p) * log(1 - p, 2)


    def information_gain(self, X, y, feature, threshold):
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        H_y = self.entropy(y)
        p = len(X[X[feature] >= threshold]) / len(X)
        if len(y_left) != 0:
            p_left = self.entropy(y_left)
        else:
            p_left = 0
        if len(y_right) != 0:
            p_right = self.entropy(y_right)
        else:
            p_right = 0
        H_y_given_x = p * p_right + (1 - p) * p_left
        return H_y - H_y_given_x

    def best_split(self, X, y):
        features = list(X.columns.values)
        random.shuffle(features)
        best_information_gain = 0
        best_feature = None
        best_threshold = None
        for feature in features:
            thresholds = list(set(list(X[feature])))
            for threshold in thresholds:
                info_gain = self.information_gain(X, y, feature, threshold)
                if info_gain >= best_information_gain:
                    best_information_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if self.is_splitting_finished(depth, len(X.columns), len(X)):
            return None

        best_feature, best_threshold = self.best_split(X, y)
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)

        left_node = self.build_tree(X_left, y_left, depth=depth + 1)
        right_node = self.build_tree(X_right, y_right, depth=depth + 1)

        value = None
        if left_node is None or right_node is None:
            true_value = len(y[y['target'] == 1])
            false_value = len(y[y['target'] == 0])
            if true_value >= false_value:
                value = 1
            else:
                value = 0

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node, value=value)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        tree = self.root
        predicted_value = []
        for index in list(X.index):
            data = X.loc[index]
            current_tree = tree
            for depth in range(self.max_depth):
                if Node.is_leaf(current_tree):
                    predicted_value.append(current_tree.value)
                    break
                feature = current_tree.feature
                threshold = current_tree.threshold
                if data[feature] <= threshold:
                    current_tree = current_tree.left
                if data[feature] > threshold:
                    current_tree = current_tree.right

        return predicted_value

breast_cancer_pdf = pd.read_csv("breast_cancer.csv")

x_train, x_val, y_train, y_val = train_test_split(breast_cancer_pdf.drop('target', axis=1), breast_cancer_pdf[['target']], test_size=0.70, random_state=42)

# clf = DecisionTree(4, 4)
# clf.fit(x_train, y_train)
# y_val_pred = clf.predict(x_val)
# result = []
# for y_p, y_t in zip(y_val_pred, list(y_val['target'])):
#     if y_p == y_t:
#         result.append(1)
#     else:
#         result.append(0)
# accuracy = sum(result) / len(result)
# print(accuracy)

max_depths = [1, 3, 4]
min_samples_splits = [1, 2, 3]


best_max_depth = None
best_min_samples_split = None
best_accuracy = 0
best_model = None
for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        clf = DecisionTree(max_depth, min_samples_split)
        clf.fit(x_train, y_train)
        y_val_pred = clf.predict(x_val)
        result = []
        for y_p, y_t in zip(y_val_pred, list(y_val['target'])):
            if y_p == y_t:
                result.append(1)
            else:
                result.append(0)
        accuracy = sum(result) / len(result)
        # print(f"min={min_samples_split}__depth={max_depth}__acu={accuracy}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_max_depth = max_depth
            best_min_samples_split = min_samples_split
            best_model = clf

x_test = pd.read_csv("test.csv")

y_test_pred = best_model.predict(x_test)
pd.DataFrame(y_test_pred, columns=['target']).to_csv('output.csv', index=False)

