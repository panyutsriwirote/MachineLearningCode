from typing import Callable

Hypothesis = Callable[[float], float]

class LinearRegressor:
    """
    y = ax + b
    """
    def __init__(self, data: list[tuple[float, float]], num_step: int, learning_rate: float = 0.1):
        n = len(data)
        a, b = 0, 0
        h: Hypothesis = lambda x: a * x + b
        for step in range(1, num_step + 1):
            loss = sum((h(x) - y) ** 2 for x, y in data) / n
            print(f"Step {step}")
            print(f"h: y = {a}x + {b}")
            print(f"loss: {loss}\n")
            a_gradient = sum((h(x) - y) * x for x, y in data) / n
            b_gradient = sum(h(x) - y for x, y in data) / n
            new_a = a - learning_rate * a_gradient
            new_b = b - learning_rate * b_gradient
            a, b = new_a, new_b

if __name__ == "__main__":
    LinearRegressor([(1, 5), (2, 7), (3, 9)], 1000)
