import math
import numpy as np
import random
class Tensor:
    def __init__(self,data,_children=(),_op="",label=''):
        self.data=np.asarray(data,dtype=np.float64)
        self._prev=set(_children)
        self.grad=np.zeros_like(self.data)
        self._backward=lambda:None
        self._op=_op
        self.label=label

    def __repr__(self):
        return f"Tenser(shape={self.data.shape},data={self.data})"
    
    def _handle_broadcast(self,orig_data,grad):
        #print(f"orig_data.shape: {orig_data.shape}, grad.shape: {grad.shape}")
        if orig_data.shape==grad.shape:
            return grad
        axes_to_sum=[]
        orig_dim=orig_data.ndim
        grad_dim=grad.shape
        diff=len(grad_dim)-orig_dim
        for i in range(diff):
            axes_to_sum.append(i)
        for i in range(orig_dim):
            if orig_data.shape[i]==1 and grad.shape[i+diff]>1:
                axes_to_sum.append(i+diff)
        if axes_to_sum:
            grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
            if diff > 0:
                grad = np.squeeze(grad, axis=tuple(range(diff)))
        return grad

    
    def __add__(self,other):
        other=other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(self.data+other.data,(self,other),"+")
        def _backward():
            self.grad+=self._handle_broadcast(self.data, out.grad)
            other.grad+=self._handle_broadcast(other.data, out.grad)
        out._backward=_backward
        return out
    
    def __radd__(self, other): 
        return self + other

    def __mul__(self,other):
        other=other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad+=self._handle_broadcast(self.data, other.data*out.grad)
            other.grad+=self._handle_broadcast(other.data, self.data*out.grad)
        out._backward=_backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")
        def _backward():
            #print(f"self.shape={self.data.shape}, other.shape={other.data.shape}")
            if self.data.ndim == 1 and other.data.ndim == 1:
                self.grad+=out.grad * other.data
                other.grad+=self.data*out.grad
            elif other.data.ndim==1:
                self.grad+=np.outer(out.grad, other.data)
                other.grad+=self.data.T @ out.grad
            else:
                self.grad+=self._handle_broadcast(self.data,out.grad@np.swapaxes(other.data,-1,-2))
                other.grad+=self._handle_broadcast(other.data,np.swapaxes(self.data,-1,-2)@out.grad)
        out._backward=_backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def exp(self):
        x = self.data
        out = Tensor(np.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        return out
    
    def tanh(self):
        n=self.data
        t = (np.exp(2*n) - 1)/(np.exp(2*n) + 1)
        out=Tensor(t,(self,),'tanh')
        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out
    
    def relu(self):
        t = np.maximum(0, self.data)
        out=Tensor(t,(self,),"relu")
        def _backward():
            self.grad+=(self.data>0)*out.grad
        out._backward=_backward
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
        self.grad=np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
            
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other):
        return self+(-other)

    def __rsub__(self, other): 
        return other + (-self)
    
    def __rmul__(self, other):  #other* self
        return self* other
    
    def sum(self):
        out=Tensor(self.data.sum(),(self,),"sum")
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward=_backward
        return out
    
    def squeeze(self, axis=None):
        out = Tensor(np.squeeze(self.data, axis=axis), (self,), "squeeze")
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out=Tensor(self.data.T,(self,),"T")
        def _backward():
            self.grad+=out.grad.T
        out._backward=_backward
        return out
    
def stack(tensors, axis=0):
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), "stack")
    def _backward():
        for i, t in enumerate(tensors):
            t.grad += out.grad[i]
    out._backward = _backward
    return out

class Neuron:
    def __init__(self, nin,activation='tanh'):
        self.w = Tensor(np.random.uniform(-1, 1, (nin,)))
        self.b = Tensor(np.random.uniform(-1, 1, ()))
        self.activation=activation

    def __call__(self, x):
        if self.activation=='tanh':
            return (x@self.w + self.b).tanh()
        if self.activation=='relu':
            return (x@self.w+ self.b).relu()
        if self.activation=='linear':
            return x@self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class Layer:
    def __init__(self, nin, nout,activation="relu"):
        self.neurons = [Neuron(nin,activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return stack(outs, axis=0).T

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts,activation='tanh'):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], activation=activation if i < len(nouts)-1 else 'linear') for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if x.data.ndim > 1 and x.data.shape[-1] == 1:
            x = x.squeeze(axis=-1)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def MSE (ypred, ygt):
    if isinstance(ypred, list):
        return sum((yp - yt)**2 for yp, yt in zip(ypred, ygt)) * (1/len(ypred))
    else:
        ypred = ypred.squeeze() if ypred.data.ndim > 1 else ypred
        diff = ypred - ygt
        return (diff ** 2).sum() * (1 / ypred.data.shape[0])

if __name__ == "__main__":
    X = Tensor(np.array([
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0],
]))
ys = Tensor(np.array([1.0, -1.0, -1.0, 1.0]))
mlp = MLP(3, [4, 4, 1], activation='tanh')

for step in range(30):
    ypred = mlp(X)
    print("ypred:", ypred.data)
    print("ys:", ys.data)
    loss = MSE(ypred, ys)
    
    for p in mlp.parameters():
        p.grad = np.zeros_like(p.data)
    loss.backward()
    
    for p in mlp.parameters():
        p.data -= 0.05 * p.grad
    
    print(f"step {step:2d}, loss: {loss.data:.6f}")
