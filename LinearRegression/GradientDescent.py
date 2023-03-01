from typing import Callable
from random import randint

Hypothesis = Callable[[float], float]

def LinearRegression(data: list[tuple[float, float]], batch_size: int, num_epochs: int, learning_rate: float):
    """
    y = ax + b
    """
    a = b = 0
    h: Hypothesis = lambda x: a * x + b
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:\n")
        start, stop = 0, batch_size
        step_data = data[start:stop]
        step = 1
        while step_data != []:
            n = len(step_data)
            a_gradient = sum((h(x) - y) * x for x, y in step_data) / n
            b_gradient = sum(h(x) - y for x, y in step_data) / n
            a = a - learning_rate * a_gradient
            b = b - learning_rate * b_gradient
            loss = sum((h(x) - y) ** 2 for x, y in step_data) / n
            print(f"Step {step}: h is y = {a}x + {b}, loss = {loss}")
            start, stop = stop, stop + batch_size
            step_data = data[start:stop]
            step += 1
        print("")
    return h

if __name__ == "__main__":
    true_a, true_b = 2, 3
    true_h: Hypothesis = lambda x: true_a * x + true_b
    h = LinearRegression([(x, true_h(x)) for x in (randint(1, 10) for _ in range(100))], 50, 1000, 0.01)
    print(f"True h is y = {true_a}x + {true_b}")
    print("Prediction:")
    for i in range(1, 11):
        prediction = h(i)
        gold = true_h(i)
        print(f"h({i}) = {prediction}, gold = {gold}, error = {prediction - gold}")
