from typing import Callable, List, Tuple

Vector = List[float]
Matrix = List[List[float]]

A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 2], [3, 4], [5, 6]]

def shape(a: Matrix) -> Tuple[int, int]:
    n_rows = len(a)
    n_cols = len(a[0]) if a else 0
    return n_rows, n_cols

assert shape(A) == (2, 3)

def get_row(a: Matrix, i: int) -> Vector:
    return a[i]

def get_col(a: Matrix, j: int) -> Vector:
    return [a_i[j] for a_i in a]

def make_matrix(n_rows: int, n_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j) for j in range(n_cols)] for i in range(n_rows)]

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]]
