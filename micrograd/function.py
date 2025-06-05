from micrograd.engine import Value


def mse_loss(input: list[Value], target: list[float]) -> Value:
    loss = sum((yprd - ygt) ** 2 for yprd, ygt in zip(input, target))
    loss = loss / len(target)

    return loss


def relu(input: Value) -> Value:
    out = Value(input.data if input.data > 0 else 0, (input, ), 'ReLu')

    def _backward():
        input.grad = (input.data > 0) * out.grad

    out._backward = _backward

    return out
