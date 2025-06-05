import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    products = x[1:] * x[:-1]
    valid_products = products[(x[1:] % 3 == 0) | (x[:-1] % 3 == 0)]
    if valid_products.size > 0:
        return valid_products.max()
    return -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.dot(image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_decoded = np.repeat(x[:, 0], x[:, 1])
    y_decoded = np.repeat(y[:, 0], y[:, 1])
    if len(x_decoded) == len(y_decoded):
        return np.dot(x_decoded, y_decoded)
    return -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    matrix = X.dot(Y.T)
    norm = np.dot(norm_X, norm_Y.T)
    return np.where(norm == 0, 1, matrix/norm)