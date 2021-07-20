import math
import random
import matplotlib.pyplot as plt
from collections import Counter

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))

def binomial_histogram(p: float, n: int, n_pts: int) -> None:
    data = [binomial(n, p) for _ in range(n_pts)]
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
                [v / n_pts for v in histogram.values()],
                0.8,
                color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title('binomial distribution vs. normal approximation')
    plt.show()

binomial_histogram(0.75, 100, 10000)
