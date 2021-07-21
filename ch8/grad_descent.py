from typing import Callable
import matplotlib.pyplot as plt

def diff_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [diff_quotient(square, x, h=0.001) for x in xs]

plt.title('Actual Derivatives vs. Estimates')
plt.plot(xs, actuals, 'rx', label = 'Actual')
plt.plot(xs, estimates, 'b+', label = 'Estimate')
plt.legend(loc = 9)
plt.show()
