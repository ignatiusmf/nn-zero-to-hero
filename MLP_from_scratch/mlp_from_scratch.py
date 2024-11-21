import math
import numpy as np
import matplotlib.pyplot as plt

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
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():
      self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
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
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

import random 
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
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
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



def train():
    global losses
    global steps_total 
    steps_total = steps_total + 1

    ypred = [n(x) for x in xtrain]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ytrain, ypred)])/len(ypred) 

    print(steps_total, 'loss data', loss.data) 

    for p in n.parameters():
        p.grad = 0.0

    loss.backward() 
    losses.append(loss.data)

    for p in n.parameters():
        step_size = 0.1 if steps_total < 50 else 0.01
        p.data += -step_size*p.grad



def train_and_visualize_loss(train_steps,speed):
    global losses
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    steps = np.arange(train_steps) 
    losses = []


    line1, = ax.plot([], [])

    def update(frame):
        train()
        line1.set_data(steps[:frame],losses[:frame])
        ax.relim()       # Recalculate limits based on data
        ax.autoscale_view()

    ani = FuncAnimation(fig, update, frames=train_steps,interval=speed,repeat=False)
    plt.show()


def visualize_training(train_steps,speed):
    global losses
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    xaxis =  [[x] for x in np.arange(0,7,0.25)]

    yaxisanim = [n(x).data for x in xaxis]  


    line1, = ax.plot([], [])
    line2, = ax.plot(xtest,ygt)

    def update(frame):
        train()
        yaxisanim = [n(x).data for x in xaxis]  
        line1.set_data(xaxis, yaxisanim)
        ax.relim()       # Recalculate limits based on data
        ax.autoscale_view()

    ani = FuncAnimation(fig, update, frames=train_steps,interval=speed,repeat=False)
    plt.show()


steps_total = 0

random.seed(3) 
n = MLP(1,[20,20,20,1])

xtrain = [[random.uniform(0,7)] for _ in range(100)]
#xs = [[-10.0], [-9.0],  [-8.0],  [-7.0],  [-6.0],  [-5.0], [-4.0],  [-3.0],  [-2.0],  [-1.0],   [0.0],   [1.0],   [2.0],   [3.0],   [4.0],   [5.0],   [6.0],   [7.0],   [8.0],   [9.0],  [10.0]]

def y_values(xs):
    ys = []
    for i in range(len(xs)):
        ys.append(math.sin(xs[i][0]))
    return(ys)
ytrain = y_values(xtrain)

print(len(xtrain),len(ytrain))


xtest = [[x] for x in np.arange(0,7,0.25)]
yfirst = [n(x).data for x in xtest]
ygt = y_values(xtest)

train_and_visualize_loss(1000,0)
visualize_training(1000,0)

ytest = [n(x).data for x in xtest]




plt.plot(xtest, yfirst)
plt.plot(xtest,ytest)
plt.plot(xtest, ygt)

plt.show()


















