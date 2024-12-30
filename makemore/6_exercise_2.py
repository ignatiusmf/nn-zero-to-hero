from pprint import pprint
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import os
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)
torch.set_default_device('cuda')

words = open(os.path.dirname(os.path.abspath(__file__)) +
             '/names.txt').read().split()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for c in w + '.':
            X.append(context)
            Y.append(stoi[c])
            # print(''.join(itos[con] for con in context), '-->', c )
            context = context[1:] + [stoi[c]]


    X, Y = torch.tensor(X), torch.tensor(Y)
    return X, Y


n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
random.shuffle(words)

Xtr, Ytr = build_dataset(words[:n1])
Xde, Yde = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) * 5/3/fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xnorm = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = xnorm * self.gamma + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * \
                    (1-self.momentum) + xmean * self.momentum
                self.running_var = self.running_var * \
                    (1-self.momentum) + xvar * self.momentum
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


vocab_size = len(chars) + 1

n_emb = 3
n_hidden = 100

C = torch.randn((vocab_size, n_emb))
layers = [
    Linear(n_emb*block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
]

with torch.no_grad():
    layers[-1].weight *= 1


parameters = [C] + [p for layer in layers for p in layer.parameters()]
# print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


batch_size = 32
lr = 0.51
losses = []
epochs = 100
ud = []

for epoch in range(epochs):
    batch = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[batch], Ytr[batch]

    emb = C[Xb].view(-1, n_emb*block_size)
    x = emb.view(-1, n_emb*block_size)
    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, Yb)

    for layer in layers:
        layer.out.retain_grad()  # AFTER_DEBUG: would take out retain_graph
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad

    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item()
                  for p in parameters])

    losses.append(loss.log10().item())

    if epoch % (epochs/10) == 0 or epoch == epochs - 1:
        print(f'{epoch:7d} {loss.item():.6f}')

    # if epoch == epochs * 0.8:
    #     print('\nAdjusting learning rate and batch size\n')
    #     batch_size *= 2
    #     lr /= 5

    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item()
                  for p in parameters])


def evaluate():
    print(f'\n{'-'*55}\nLoss function\n{'-'*55}\n')
    emb = C[Xtr]
    x = emb.view(-1, n_emb*block_size)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr)

    print(f'{loss:.6f} - DEV LOSS')


def loss_graph():
    moving_average = int(epochs**0.5)
    plt.plot(np.convolve(losses, np.ones(
        moving_average)/moving_average, mode='valid'))
    plt.show()


def sample(num):
    print(f'\n{'-'*55}\nSAMPLING\n{'-'*55}\n')
    names = []
    for i in range(num):
        out = ''
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            x = emb.view(-1, block_size*n_emb)
            for layer in layers:
                x = layer(x)

            logits = x
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, 1, replacement=True).item()
            if ix == 0:
                break

            out += itos[ix]
            context = context[1:] + [ix]
        names.append(out)
    return names


def tanh_values():
    print(f'\n{'-'*55}\nData distribution of Tanh layers\n{'-'*55}\n')

    legends = []
    for i, layer in enumerate(layers):
        if isinstance(layer, Tanh):
            t = layer.out.cpu().detach()
            print(f'Layer {i} {layer.__class__.__name__} {tuple(t.shape)}, std {t.std():.2f} mean {
                  t.mean():.2f}, saturated {(t.abs() > 0.97).float().mean()*100:.2f}%')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1], hy)
            legends.append(f'layer {i} {layer.__class__.__name__} {
                           tuple(t.shape)}')
    plt.legend(legends)
    plt.title('Tanh data distribution')
    plt.show()


def tanh_grads():
    print(f'\n{'-'*55}\nGradient values of tanh layers\n{'-'*55}\n')

    legends = []
    for i, layer in enumerate(layers):
        if isinstance(layer, Tanh):
            t = layer.out.grad.cpu().detach()
            print(f'{layer.__class__.__name__} std {
                  t.std().item():.3f} mean {t.mean().item():.3f}')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1], hy)

            legends.append(f'layer {i} {layer.__class__.__name__} {
                           tuple(t.shape)}')

    plt.legend(legends)
    plt.title("Tanh gradients distribution")
    plt.show()

# tanh_grad_distribution()
def weight_grads():
    print(f'\n{'-'*55}\nGradient values of weights in linear layers\n{'-'*55}\n')
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            t = p.grad.cpu().detach()
            print(f'Shape {tuple(p.shape)}, mean {t.mean():.3f} std {t.std():.3f}',
                f'grad:data mean {(t.mean() / p.mean()):.3e} grad:data {(t.std() / p.std()):.3e}')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1], hy)
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title("Weight gradients of linear layers")
    plt.show()

# weight_grads()
exit()


print(f'\n{'-'*55}\nUpdate rate\n{'-'*55}\n')

legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
# these ratios should be ~1e-3, indicate on plot
plt.plot([0, len(ud)], [-3, -3], 'k')
plt.legend(legends)
plt.show()
