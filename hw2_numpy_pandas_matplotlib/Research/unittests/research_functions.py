from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if len(x) != len(y):
        return False
    return sorted(x) == sorted(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    products = []
    for i in range(len(x) - 1):
        product = x[i] * x[i + 1]
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            products.append(product)
    if products:
        return max(products)
    return -1



def scalar_prod(x, y):
    sum = 0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    result_image = []
    ind = 0
    for el in image:
        result_image.append([])
        result_image[ind] = [scalar_prod(item, weights) for item in el] 
        ind += 1
    return result_image

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_decoded = []
    y_decoded = []
    for i in x:
        for j in range(i[1]):
            x_decoded.append(i[0])
    for i in y:
        for j in range(i[1]):
            y_decoded.append(i[0])      
    if len(x_decoded) != len(y_decoded):
        return -1 
    sum = 0
    for i in range(len(x_decoded)):
        sum += x_decoded[i] * y_decoded[i]
    return sum

def cosinus(x, y):
    num = sum(x_i * y_i for x_i, y_i in zip(x, y))
    norm_x = sum(elem * elem for elem in x) ** 0.5
    norm_y = sum(elem * elem for elem in y) ** 0.5
    if norm_x == 0 or norm_y == 0:
        return 1
    return num / (norm_x * norm_y)

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    mas = [[0] * len(Y) for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            mas[i][j] = cosinus(X[i], Y[j])
    return mas