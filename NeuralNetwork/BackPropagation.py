from functools import partial
from random import uniform
from math import exp
from typing import Callable
from tqdm import tqdm

class Network:

    Function = Callable[[float], float]
    # activation functions and their derivatives
    activation: dict[str, tuple[Function, Function]] = {
        "sigmoid": (lambda x: 1 / (1 + exp(-x)), lambda o: o * (1 - o))
    }
    random_weight = partial(uniform, -1, 1)

    def __init__(self, input_len: int, hidden_len: int, output_len: int, activation_func_name: str):
        self.structure = (input_len, hidden_len, output_len)
        self.activate, self.derivative = Network.activation[activation_func_name]
        self.weights = (
            [[Network.random_weight() for _ in range(input_len + 1)] for _ in range(hidden_len)], # hidden weight
            [[Network.random_weight() for _ in range(hidden_len + 1)] for _ in range(output_len)] # output weight
        )
        self.inter_values = (
            [0.0] * hidden_len + [1], # hidden o
            [0.0] * output_len # output o
        )
        self.deltas = (
            [0.0] * hidden_len, # hidden delta
            [0.0] * output_len # output delta
        )
        self.gradients = (
            [[0.0] * (input_len + 1) for _ in range(hidden_len)], # hidden gradient
            [[0.0] * (hidden_len + 1) for _ in range(output_len)] # output gradient
        )

    def reset_gradients(self):
        input_len, hidden_len, output_len = self.structure
        for i in range(output_len):
            for j in range(hidden_len + 1):
                self.gradients[1][i][j] = 0
        for i in range(hidden_len):
            for j in range(input_len + 1):
                self.gradients[0][i][j] = 0

    def calculate(self, input_values: list[float]):
        input_len, hidden_len, output_len = self.structure
        assert input_len == len(input_values),\
            f"Incompatible input length. This Network expects {input_len} input values but gets {len(input_values)}."
        # hidden layer
        for i in range(hidden_len):
            self.inter_values[0][i] = self.activate(sum(weight * value for weight, value in zip(self.weights[0][i], input_values + [1])))
        # output layer
        for i in range(output_len):
            self.inter_values[1][i] = self.activate(sum(weight * value for weight, value in zip(self.weights[1][i], self.inter_values[0])))

    def learn(self, training_examples: list[tuple[list[float], list[float]]], batch_size: int, num_epochs: int, learning_rate: float):
        input_len, hidden_len, output_len = self.structure
        assert all(len(inputs) == input_len and len(outputs) == output_len for inputs, outputs in training_examples),\
            f"Incompatible training examples. Make sure that every example has {input_len} input values and {output_len} output values."
        for _ in tqdm(range(num_epochs)):
            start, stop = 0, batch_size
            step_examples = training_examples[start:stop]
            while step_examples != []:
                self.reset_gradients()
                for input_values, target_outputs in step_examples:
                    self.calculate(input_values)
                    # output delta
                    for i, (o, t) in enumerate(zip(self.inter_values[1], target_outputs)):
                        self.deltas[1][i] = self.derivative(o) * (t - o)
                    # hidden delta
                    for i, o in enumerate(self.inter_values[0][:-1]):
                        self.deltas[0][i] = self.derivative(o) * sum(self.weights[1][k][i] * self.deltas[1][k] for k in range(output_len))
                    # update gradients
                    for i in range(output_len):
                        for j in range(hidden_len + 1):
                            self.gradients[1][i][j] += self.deltas[1][i] * self.inter_values[0][j]
                    for i in range(hidden_len):
                        for j, x in enumerate(input_values + [1]):
                            self.gradients[0][i][j] += self.deltas[0][i] * x
                # update weights
                num_examples = len(step_examples)
                for i in range(output_len):
                    for j in range(hidden_len + 1):
                        self.weights[1][i][j] += learning_rate * (self.gradients[1][i][j] / num_examples)
                for i in range(hidden_len):
                    for j in range(input_len + 1):
                        self.weights[0][i][j] += learning_rate * (self.gradients[0][i][j] / num_examples)
                start, stop = stop, stop + batch_size
                step_examples = training_examples[start:stop]

    def predict(self, input_values: list[float]):
        self.calculate(input_values)
        return self.inter_values[1]

if __name__ == "__main__":

    print("XOR")
    XOR = [
        ([1.0,1.0],[0.0]),
        ([1.0,0.0],[1.0]),
        ([0.0,1.0],[1.0]),
        ([0.0,0.0],[0.0])
    ]
    network = Network(2, 2, 1, "sigmoid")
    network.learn(XOR, batch_size=4, num_epochs=10000, learning_rate=10)
    prediction = [1 if x > 0.5 else 0 for x in network.predict([1,1])]
    print(f"1 XOR 1 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([1,0])]
    print(f"1 XOR 0 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0,1])]
    print(f"0 XOR 1 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0,0])]
    print(f"0 XOR 0 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")

    print("8-BIT IDENTITY")
    IDENTITY = [
        ([1.0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1.0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 1.0, 0], [0, 0, 0, 0, 0, 0, 1.0, 0]),
        ([0, 0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, 0, 1.0]),
    ]
    network = Network(8, 2, 8, "sigmoid")
    network.learn(IDENTITY, batch_size=8, num_epochs=10000, learning_rate=10)
    prediction = [1 if x > 0.5 else 0 for x in network.predict([1.0, 0, 0, 0, 0, 0, 0, 0])]
    print(f"1 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 1.0, 0, 0, 0, 0, 0, 0])]
    print(f"2 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 1.0, 0, 0, 0, 0, 0])]
    print(f"3 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 0, 1.0, 0, 0, 0, 0])]
    print(f"4 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 0, 0, 1.0, 0, 0, 0])]
    print(f"5 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 0, 0, 0, 1.0, 0, 0])]
    print(f"6 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 0, 0, 0, 0, 1.0, 0])]
    print(f"7 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
    prediction = [1 if x > 0.5 else 0 for x in network.predict([0, 0, 0, 0, 0, 0, 0, 1.0])]
    print(f"8 -> {[round(x, 2) for x in network.inter_values[0][:-1]]} -> {prediction}")
