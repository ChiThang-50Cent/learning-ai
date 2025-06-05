import random
from typing import Union

import micrograd.function as F
from micrograd.engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list:
        return []


class Neuron(Module):
    def __init__(
        self,
        nin: int,
    ):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x: list) -> Value:
        out = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
    ):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Union[float, int]]) -> Union[list[Value], Value]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [w for neuron in self.neurons for w in neuron.parameters()]


class MLP(Module):
    def __init__(
        self,
        nin: int,
        nouts: list[int],
    ):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(
        self,
        x: list[Union[float, int]],
    ) -> Union[list[Value], Value]:
        relu = ReLu()
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = relu(x)

        x = self.layers[-1](x)

        return x

    def parameters(self):
        return [w for layer in self.layers for w in layer.parameters()]


class MSELoss:
    def __call__(
        self,
        input: list[Value],
        target: list[Union[float, int]],
    ) -> Value:
        return F.mse_loss(input, target)


class ReLu:
    def __call__(
        self,
        input: list[Value],
    ):
        out = (
            F.relu(input)
            if isinstance(input, Value)
            else [F.relu(i) for i in input]
        )
        return out
