import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# Configuration ###
random.seed(0)
torch.manual_seed(0)
torch.set_default_device("cuda")

# Loading data ###
words = open(os.path.dirname(os.path.abspath(__file__)) +
             '/names.txt', 'r').read().splitlines()
chars = sorted(set(''.join(words)))

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


# Creating datasets
block_size = 5


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for c in w + '.':
            X.append(context)
            Y.append(stoi[c])
            # print(''.join(itos[i] for i in context), '--->', c)
            context = context[1:] + [stoi[c]]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
random.shuffle(words)
Xtr, Ytr = build_dataset(words[:n1])
Xde, Yde = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

embed_size = 3
vocab_size = len(chars) + 1
n_hidden = 200

C = torch.randn((vocab_size, embed_size))
W = torch.randn((embed_size*block_size, n_hidden)) * \
    5/3/(embed_size*block_size)**0.5
O = torch.randn((n_hidden, vocab_size)) * 0.01
o = torch.zeros((vocab_size))

bn_scale = torch.ones((1, n_hidden))
bn_shift = torch.zeros((1, n_hidden))

bnstd_running = torch.ones((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))

parameters = [C, W, bn_shift, bn_scale, O, o]
for p in parameters:
    p.requires_grad = True

batch_size = 32

epochs = 10000
lr = 0.1
losses = []
for epoch in range(epochs):

    batch = torch.randint(0, Xtr.shape[0], (batch_size,))

    emb = C[Xtr[batch]].view(-1, block_size*embed_size)

    hpreact = emb @ W

    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
    with torch.no_grad():
        bnstd_running = bnstd_running * 0.999 + bnstd * 0.001
        bnmean_running = bnmean_running * 0.999 + bnmean * 0.001

    hpreact = bn_scale * (hpreact - bnmean) / bnstd - bn_shift

    h = torch.tanh(hpreact)

    logits = h @ O + o
    loss = F.cross_entropy(logits, Ytr[batch])

    for p in parameters:
        p.grad = None

    loss.backward()
    losses.append(loss.log10().item())

    for p in parameters:
        p.data += -lr * p.grad

    if epoch == epochs * 0.8:
        lr = 0.01
        batch_size = batch_size * 2
        print("\nAdjusting training\n")

    if epoch % (epochs/10) == 0:
        print(epoch, loss.item())


plt.hist(hpreact.view(-1).tolist(), 50)
plt.show()
plt.hist(h.view(-1).tolist(), 50)
plt.show()


def evaluate():
    emb = C[Xde].view(-1, block_size*embed_size)
    hpre = emb @ W
    bnhpre = bn_scale * (hpre - bnmean_running) / bnstd_running - bn_shift
    h = torch.tanh(bnhpre)
    logits = h @ O + o
    loss = F.cross_entropy(logits, Yde)
    print(f"\nDev loss - {loss.item()}\n")


evaluate()
if epochs > 25:
    moving_average = int(batch_size*math.log10(epochs))  # int(epochs/10)
    plt.plot(np.convolve(losses, np.ones(moving_average)/moving_average, mode='valid'))
else:
    plt.plot(losses)

plt.show()


for i in range(10):
    context = torch.tensor([[0] * block_size])
    name = []
    while True:
        emb = C[context].view(-1, block_size*embed_size)
        hpreact = emb @ W
        bnhpre = bn_scale * (hpreact - bnmean_running) / \
            bnstd_running - bn_shift
        h = torch.tanh(bnhpre)
        logits = h @ O + o

        probs = F.softmax(logits, dim=1)

        next = torch.multinomial(probs, 1, replacement=True).item()
        if next == 0:
            break

        name.append(itos[next])

        context = torch.cat((context[:, 1:], torch.tensor([[next]])), dim=1)

    print(i, ''.join(name))
