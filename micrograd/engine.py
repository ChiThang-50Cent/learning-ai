from typing import Union


class Value:
    def __init__(
        self, data: Union[int, float], _children: tuple = (), _op: str = ""
    ):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        if not isinstance(other, (int, float)):
            raise TypeError("Pow only support int or float")

        out = Value(self.data**other, (self,), "**")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # def relu(self):
    #     out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    #     def _backward():
    #         self.grad += (out.data > 0) * out.grad
    #     out._backward = _backward

    #     return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(root: "Value"):
            if root not in visited:
                visited.add(root)
                for node in root._prev:
                    build_topo(node)
                topo.append(root)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: Union[int, float]) -> "Value":
        return self + other

    def __rmul__(self, other: Union[int, float]) -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> 'Value':
        return other + (-self)

    def __truediv__(self, other: Union[int, float, "Value"]) -> "Value":
        return self * (other**-1)

    def __rtruediv__(self, other: Union[int, float, "Value"]) -> "Value":
        return other * (self**-1)


if __name__ == "__main__":
    a = Value(5)
    b = Value(10)

    L = a / b

    L.backward()

    print(L)
    print(a)
    print(b)
