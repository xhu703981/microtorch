import math
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="", label=''):
        self.data = np.asarray(data, dtype=np.float64)
        self._prev = set(_children)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"

    def _handle_broadcast(self, orig_data, grad):
        if orig_data.shape == grad.shape:
            return grad
        axes_to_sum = []
        orig_dim = orig_data.ndim
        grad_dim = grad.shape
        diff = len(grad_dim) - orig_dim
        for i in range(diff):
            axes_to_sum.append(i)
        for i in range(orig_dim):
            if orig_data.shape[i] == 1 and grad.shape[i + diff] > 1:
                axes_to_sum.append(i + diff)
        if axes_to_sum:
            grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
            if diff > 0:
                grad = np.squeeze(grad, axis=tuple(range(diff)))
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += self._handle_broadcast(self.data, out.grad)
            other.grad += self._handle_broadcast(other.data, out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += self._handle_broadcast(self.data, other.data * out.grad)
            other.grad += self._handle_broadcast(other.data, self.data * out.grad)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")
        def _backward():
            if self.data.ndim == 1 and other.data.ndim == 1:
                # 1D dot product
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            else:
                # 2D or batch matmul
                self.grad += self._handle_broadcast(self.data, out.grad @ np.swapaxes(other.data, -1, -2))
                other.grad += self._handle_broadcast(other.data, np.swapaxes(self.data, -1, -2) @ out.grad)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def stack(tensors, axis=0):
    """Stack a list of Tensors along a new axis, with proper backward support."""
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), "stack")
    def _backward():
        for i, t in enumerate(tensors):
            t.grad += out.grad[i]
    out._backward = _backward
    return out

class Neuron:
    def __init__(self, nin):
        self.w = Tensor(np.random.uniform(-1, 1, (nin,)))
        self.b = Tensor(np.random.uniform(-1, 1, ()))

    def __call__(self, x):
        return (self.w @ x + self.b).tanh()

    def parameters(self):
        return [self.w, self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return stack(outs, axis=0)

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


if __name__ == "__main__":
    xs = [
        Tensor(np.array([2.0,  3.0, -1.0])),
        Tensor(np.array([3.0, -1.0,  0.5])),
        Tensor(np.array([0.5,  1.0,  1.0])),
        Tensor(np.array([1.0,  1.0, -1.0])),
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP(3, [4, 4, 1])

    for step in range(30):
        ypred = [mlp(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
        for p in mlp.parameters():
            p.grad = np.zeros_like(p.data)
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.01 * p.grad
        print(f"step {step:2d}, loss: {loss.data:.6f}")
