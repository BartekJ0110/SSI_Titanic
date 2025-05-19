import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class Split:
    feature_index: int
    threshold: float
    data_left: np.ndarray
    data_right: np.ndarray
    gain: float


@dataclass
class TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    label: Optional[int] = None


class SoftClassifier:
    """
    A soft classifier that transforms features based on statistical thresholds
    and assigns labels by matching samples to the closest class profile.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.feature_stats_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Calculates mean and standard deviation of each feature per class.
        """
        self.classes_ = np.unique(y)
        self.feature_stats_ = {}

        for class_ in self.classes_:
            X_class = X[y == class_]
            means = np.mean(X_class, axis=0)
            stds = np.std(X_class, axis=0)
            self.feature_stats_[class_] = {"mean": means, "std": stds}

        return self

    def _transform_features(self, X, class_):
        """
        Transforms features to binary indicators based on proximity to class means.
        """
        means = self.feature_stats_[class_]["mean"]
        stds = self.feature_stats_[class_]["std"]
        transformed = (X >= (means - self.threshold * stds)).astype(int)
        return transformed

    def predict(self, X):
        """
        Predicts class labels based on soft matching of features to class profiles.
        """
        scores = []
        for class_ in self.classes_:
            transformed = self._transform_features(X, class_)
            score = np.sum(transformed, axis=1)
            scores.append(score)

        scores = np.array(scores).T
        y_pred = self.classes_[np.argmax(scores, axis=1)]
        return y_pred

    def score(self, X, y):
        """
        Computes the accuracy of the soft classifier.
        """
        return np.sum(self.predict(X) == y) / len(X)


class DecisionTree:
    """
    A decision tree classifier with an optional stopping criterion based on soft classification.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 4,
        min_samples_split: int = 2,
        soft_classifier: Optional[SoftClassifier] = None,
        soft_threshold: float = 0.9,
    ) -> None:
        """
        Initializes the decision tree classifier.

        Args:
            criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split a node.
            soft_classifier (SoftClassifier): Optional soft classifier to evaluate stopping condition.
            soft_threshold (float): Minimum soft accuracy to stop splitting.
        """
        assert criterion in [
            "gini",
            "entropy",
        ], f"Invalid criterion: {criterion}. Choose 'gini' or 'entropy'."

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.soft_classifier = soft_classifier
        self.soft_threshold = soft_threshold

    def _entropy(self, probabilities: np.ndarray) -> float:
        """
        Computes the entropy of a set of probabilities.

        Args:
            probabilities (np.ndarray): Array of probabilities for each class.

        Returns:
            float: The entropy value.
        """
        probabilities = probabilities[probabilities > 0]
        return np.sum([-p * np.log2(p) for p in probabilities])

    def _gini(self, probabilities: np.ndarray) -> float:
        """
        Computes the Gini impurity of a set of probabilities.

        Args:
            probabilities (np.ndarray): Array of probabilities for each class.

        Returns:
            float: The Gini impurity value.
        """
        return 1 - np.sum([p**2 for p in probabilities])

    def _get_probabilities(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the probabilities for each class label in the dataset.

        Args:
            y (np.ndarray): Array of class labels.

        Returns:
            np.ndarray: Array of probabilities for each class.
        """
        labels = np.unique(y)
        probabilities = np.empty(labels.shape)

        for i, label in enumerate(labels):
            probabilities[i] = len(y[y == label]) / len(y)

        return probabilities

    def _calculate_gain(
        self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray
    ) -> float:
        """
        Calculates the information gain from a split.

        Args:
            parent (np.ndarray): The parent node's data.
            left_child (np.ndarray): The left child node's data.
            right_child (np.ndarray): The right child node's data.

        Returns:
            float: The calculated information gain.
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        parent_probabilities = self._get_probabilities(parent)
        left_probabilities = self._get_probabilities(left_child)
        right_probabilities = self._get_probabilities(right_child)

        if self.criterion == "gini":
            return self._gini(parent_probabilities) - (
                weight_left * self._gini(left_probabilities)
                + weight_right * self._gini(right_probabilities)
            )
        else:
            return self._entropy(parent_probabilities) - (
                weight_left * self._entropy(left_probabilities)
                + weight_right * self._entropy(right_probabilities)
            )

    def _split(
        self, data: np.ndarray, feature_index: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataset into two subsets based on the threshold of a feature.

        Args:
            data (np.ndarray): The dataset to split.
            feature_index (int): The index of the feature to split on.
            threshold (float): The value used to split the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The left and right splits of the dataset.
        """
        mask = data[:, feature_index] < threshold
        return data[mask, :], data[~mask, :]

    def _get_best_split(self, data: np.ndarray) -> Split:
        """
        Finds the best split for the dataset based on the criterion.

        Args:
            data (np.ndarray): The dataset to find the best split for.

        Returns:
            Split: The best split for the dataset.
        """
        best_gain = float("-inf")
        best_split = None

        for feature_index in range(self.n_features_in_):
            thresholds = np.unique(data[:, feature_index])

            for threshold in thresholds:
                data_left, data_right = self._split(data, feature_index, threshold)

                if data_left.size > 0 and data_right.size > 0:
                    y = data[:, -1]
                    y_left = data_left[:, -1]
                    y_right = data_right[:, -1]

                    gain = self._calculate_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_split = Split(
                            feature_index=feature_index,
                            threshold=threshold,
                            data_left=data_left,
                            data_right=data_right,
                            gain=gain,
                        )

        if best_split is None:
            return Split(
                feature_index=0,
                threshold=0,
                data_left=data,
                data_right=data,
                gain=0.0,
            )
        return best_split

    def _get_most_frequent_label(self, y: np.ndarray) -> int:
        """
        Returns the most frequent label in the dataset.

        Args:
            y (np.ndarray): The array of labels.

        Returns:
            int: The most frequent label.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _init_tree(self, data: np.ndarray, current_depth: int = 0) -> TreeNode:
        """
        Recursively builds the decision tree.

        Args:
            data (np.ndarray): The dataset to build the tree from.
            current_depth (int): The current depth of the tree.

        Returns:
            TreeNode: The root node of the tree.
        """
        n_samples = len(data)
        y = data[:, -1]
        X = data[:, :-1]

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self._get_best_split(data)

            if best_split.gain > 0:
                if self.soft_classifier:
                    self.soft_classifier.fit(X, y)
                    if self.soft_classifier.score(X, y) >= self.soft_threshold:
                        return TreeNode(label=self._get_most_frequent_label(y))

                left = self._init_tree(best_split.data_left, current_depth + 1)
                right = self._init_tree(best_split.data_right, current_depth + 1)

                return TreeNode(
                    feature_index=best_split.feature_index,
                    threshold=best_split.threshold,
                    left=left,
                    right=right,
                )

        return TreeNode(label=self._get_most_frequent_label(y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the decision tree on the given data.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
        """
        self.n_features_in_ = X.shape[1]
        data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
        self.tree_ = self._init_tree(data)

    def _predict(self, x: np.ndarray, node: TreeNode) -> int:
        """
        Recursively predicts the label for a given sample.

        Args:
            x (np.ndarray): The input data point.
            node (TreeNode): The current node in the tree.

        Returns:
            int: The predicted label.
        """
        if node.label is not None:
            return node.label

        if x[node.feature_index] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (n_samples,).
        """
        return np.array([self._predict(x, self.tree_) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the accuracy of the classifier.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): True class labels (n_samples,).

        Returns:
            float: Accuracy of the classifier.
        """
        return np.sum(self.predict(X) == y) / len(X)
