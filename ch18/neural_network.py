from typing import List
import math

Vector = List[float]

def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), 'vectors must be the same length'
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    calculation = dot(weights, x) + bias
    return step_function(calculation)

and_weights = [2., 2]
and_bias = -3.
assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

or_weights = [2., 2]
or_bias = -1.
assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0

not_weights = [-2.]
not_bias = 1.
assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    outputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

xor_network = [
                [[20., 20, -30],
                 [20., 20, -10]],
                [[-60., 60, -30]]]

assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

def sqerror_gradients(network: List[List[Vector]], input_vector: Vector, target_vector: Vector) -> List[List[Vector]]:
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, target_vector)]
    output_grads = [[output_deltas[i] * hidden_output for hidden_output in hidden_outputs + [1]] for i, output_neuron in enumerate(network[-1])]
    hidden_deltas = [hidden_output * (1 - hidden_output) * dot(output_deltas, [n[i] for n in network[-1]]) for i, hidden_output in enumerate(hidden_outputs)]
    hidden_grads = [[hidden_deltas[i] * input for input_vector + [1]] for i, hidden_neuron in enumerate(network[0])]
    return [hidden_grads, output_grads]
