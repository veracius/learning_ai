"""Python equivalent of nn_test.c, using Karpathy's micrograd implementation
(extracted from tutorials/2025-04-18_karpathy1/karpathy_1.ipynb).

Trains MLP(3, [4, 4, 1]) on his 4-sample dataset for 20 epochs with online SGD,
matching the C version's loop structure: per-sample zero/forward/backward/step.

Note: Karpathy's MLP keeps the bias as a separate `self.b` field; nn.h packs it
into `weights[0]` with a constant 1.0 feature. The math is equivalent. The
random initialisation uses different PRNGs (Python's Mersenne Twister vs C's
rand()), so loss curves and final predictions will not match bit-for-bit
across implementations — but should converge to similar quality on this dataset.
"""

import math
import random


# =============================================================================
# Value: scalar autograd (analogous to ugrad.h)
# =============================================================================
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# =============================================================================
# Neuron / Layer / MLP (analogous to nn.h)
# =============================================================================
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def load_weights(mlp, filename):
    """Overwrite an MLP's weights from a plain-text file (one float per line).
    Order matches nn.h's LoadMLPWeights: layer-major, neuron-major, bias first
    (Karpathy's self.b ↔ nn.h's weights[0]), then real input weights in order.
    Returns the number of values consumed.
    """
    with open(filename) as f:
        values = [float(line) for line in f if line.strip()]
    idx = 0
    for layer in mlp.layers:
        for neuron in layer.neurons:
            neuron.b.data = values[idx]; idx += 1
            for w in neuron.w:
                w.data = values[idx]; idx += 1
    return idx


# =============================================================================
# Training loop — mirrors nn_test.c: online SGD, per-sample zero/fwd/bwd/step
# =============================================================================
def main():
    xs = [
        [2.0,  3.0, -1.0],
        [3.0, -1.0,  0.5],
        [0.5,  1.0,  1.0],
        [1.0,  1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])
    load_weights(n, "init_weights.txt")  # shared init with nn_test.c for cross-implementation match
    lr = 0.05

    for epoch in range(20):
        total_loss = 0.0
        for x, y in zip(xs, ys):
            # Karpathy's MLP rebuilds the forward graph every call; the loss is
            # constructed fresh per sample, so there's no topo cache to
            # invalidate or "ZeroGrad over" — params start fresh each call.
            for p in n.parameters():
                p.grad = 0.0

            ypred = n(x)
            loss = (ypred - y) ** 2

            loss.backward()
            total_loss += loss.data

            for p in n.parameters():
                p.data -= lr * p.grad

        print(f"epoch {epoch:2d}  loss={total_loss:g}")

    print("\nFinal predictions:")
    for x, y in zip(xs, ys):
        pred = n(x)
        print(f"  x=({x[0]: .1f},{x[1]: .1f},{x[2]: .1f})  "
              f"pred={pred.data: .4f}  target={y: .1f}")


if __name__ == "__main__":
    main()
