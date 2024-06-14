import numpy as np
import unittest


def inverse_matrix_3x3(matrix):
    """
    Обчислення оберненої матриці 3x3.

    Аргументи:
    Масив з трьох масивів по 3 значення.

    Повернення:
    Масив масивів — матриця обернена до заданої 3x3 матриці
    """

    # Обчислення визначника(детермінанта) матриці
    det = np.linalg.det(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Обчислення оберненої матриці
    inverse_matrix = np.linalg.inv(matrix)
    return inverse_matrix.tolist()


# Тест клас, що наслідує від unittest.TestCase

class TestInverseMatrix3x3(unittest.TestCase):
    def test_inverse(self):
        # тестова матриця
        matrix = [
            [1, 2, 3],
            [0, 1, 4],
            [5, 6, 0]
        ]
        # очікуваний результат (інверсія матриці)
        expected_inverse = [
            [-24, 18, 5],
            [20, -15, -4],
            [-5, 4, 1]
        ]
        result = inverse_matrix_3x3(matrix)
        # Перевіряє, що отриманий результат майже дорівнює очікуваному з точністю до шести знаків після коми
        np.testing.assert_almost_equal(result, expected_inverse, decimal=6)

    # метод тестування сингулярної матриці
    def test_singular_matrix(self):
        singular_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        with self.assertRaises(ValueError):
            inverse_matrix_3x3(singular_matrix)

    # метод тестування для матриці, обернену до себе
    def test_identity_matrix(self):
        identity_matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        expected_inverse = identity_matrix
        result = inverse_matrix_3x3(identity_matrix)
        np.testing.assert_almost_equal(result, expected_inverse, decimal=6)


if __name__ == '__main__':
    unittest.main()
