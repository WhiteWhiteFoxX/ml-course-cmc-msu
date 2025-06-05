import numpy as np
import typing
from collections import defaultdict


def kfold_split(
    num_objects: int, num_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:

    indexes = np.arange(num_objects)
    fold_size = num_objects // num_folds
    excess = num_objects % num_folds
    folds = []
    start_ind = 0
    for i in range(num_folds):
        end_ind = start_ind + fold_size
        if i == num_folds - 1:
            end_ind += excess
        folds.append(indexes[start_ind:end_ind])
        start_ind = end_ind
    split_pairs = []
    for i in range(num_folds):
        training_set = np.concatenate([folds[j] for j in range(num_folds) if j != i])
        validation_set = folds[i]
        split_pairs.append((training_set, validation_set))
    return split_pairs


def knn_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, list],
    score_function: callable,
    folds: list[tuple[np.ndarray, np.ndarray]],
    knn_class: object,
) -> dict[str, float]:

    scores = {}
    n_neighbors_params = parameters["n_neighbors"]
    metrics_params = parameters["metrics"]
    weights_params = parameters["weights"]
    normalizers = parameters["normalizers"]

    for normalizer, normalizer_name in normalizers:
        for n_neighbors in n_neighbors_params:
            for metric in metrics_params:
                for weight in weights_params:
                    knn = knn_class(
                        n_neighbors=n_neighbors, metric=metric, weights=weight
                    )
                    fold_scores = []
                    for train_ind, val_ind in folds:
                        X_train, X_val = X[train_ind], X[val_ind]
                        y_train, y_val = y[train_ind], y[val_ind]

                        if normalizer:
                            normalizer.fit(X_train)
                            X_train = normalizer.transform(X_train)
                            X_val = normalizer.transform(X_val)

                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_val)

                        fold_scores.append(score_function(y_val, y_pred))

                    scores[(normalizer_name, n_neighbors, metric, weight)] = np.mean(
                        fold_scores
                    )

    return scores
