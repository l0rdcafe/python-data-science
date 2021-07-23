from typing import List, NamedTuple
from collections import Counter
import math

Vector = List[float]

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

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

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key = lambda lp: distance(lp.point, new_point))
    k_nearest_labels = [lp.labels for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)
