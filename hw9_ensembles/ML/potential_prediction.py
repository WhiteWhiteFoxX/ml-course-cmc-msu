import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np


def recenter(arr):
    non_zero_coords = np.argwhere(arr != 0)
    if non_zero_coords.size == 0:
        return arr

    min_coords, max_coords = non_zero_coords.min(axis=0), non_zero_coords.max(axis=0)
    center_non_zero = (min_coords + max_coords) // 2
    shift = (np.array(arr.shape) // 2) - center_non_zero

    new_coords = np.clip(non_zero_coords + shift, 0, np.array(arr.shape) - 1)
    result = np.zeros_like(arr)
    result[tuple(new_coords.T)] = arr[tuple(non_zero_coords.T)]

    return result


class PotentialTransformer:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([(recenter(matrix - 20)).flatten() for matrix in X])

    def fit_transform(self, X, y=None):
        return self.transform(X)


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)

    potential_transformer = PotentialTransformer()
    X_train = potential_transformer.fit_transform(X_train, Y_train)
    X_test = potential_transformer.transform(X_test)

    regressor = Pipeline(
        [
            ("vectorizer", potential_transformer),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=3000,
                    criterion="squared_error",
                    max_depth=10,
                    max_features="sqrt",
                    random_state=42,
                ),
            ),
        ]
    )

    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
