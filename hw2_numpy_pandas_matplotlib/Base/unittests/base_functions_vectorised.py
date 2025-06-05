import numpy as np


def get_part_of_array(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    return X[::4, 120:500:5]


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag_elements = np.diag(X)
    non_neg_elements = diag_elements[diag_elements >= 0]
    if non_neg_elements.size > 0:
        return np.sum(non_neg_elements)
    return -1


def replace_values(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    X_copy = np.copy(X)
    mean_vals = np.mean(X, axis=0)
    X_copy[(X > 1.5 * mean_vals) | (X < 0.25 * mean_vals)] = -1
    return X_copy