from typing import List, TypeVar, Iterator
import math
import random

Vector = List[float]

def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), 'vectors must be the same length'
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    return dot(v, v)

def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))

def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors must be the same length'
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors must be the same length'
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]

def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, 'no vectors provided'
    n_elms = len(vectors[0])
    assert all(len(v) == n_elms for v in vectors), 'vectors have different sizes'
    return [sum(vector[i] for vector in vectors) for i in range(n_elms)]

def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    print(v, gradient)
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    return [2 * error * x, 2 * error]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001
inputs = [(2, 1), (-1, 1), (0, 0), (1, 0), (0, 1)]

T = TypeVar('T')

def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    batch_starts = [start for start in range(0, len(inputs), batch_size)]
    if shuffle:
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield inputs[start:end]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size = 20):
        grade = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, 'slope should be about 20'
assert 4.9 < intercept < 5.1, 'intercept should be about 5'
