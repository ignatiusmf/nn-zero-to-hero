import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import torch.nn.functional as F

########################################################################### SETUP
torch.manual_seed(0)

start_time = time.time()

def timer(note='Time'):
    print(f'{note}: {(time.time() - start_time):.3f}')

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    print("CUDA is not available. Using CPU tensors.")

def mst(tensor, **kwargs):
    plt.matshow(tensor.detach().cpu().numpy(), **kwargs)
    plt.show()
###########################################################################3

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  
itos = {i:s for s,i in stoi.items()} 


block_size = 3 

def build_dataset(words):
    X, Y = [], [] 
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

import random
random.seed(1)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

random.shuffle(words)
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


embed_size = 2
C = torch.randn((27,embed_size))

W1 = torch.randn((block_size*embed_size,100))
b1 = torch.randn((100))

O = torch.randn((100,27))
o = torch.randn((27))

parameters = [C, W1, b1, O, o]

for p in parameters:
    p.requires_grad = True


lr = 0.1
batch_size = 320
epochs = 50000
for epoch in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    emb = C[Xtr[ix]]
    H1 = torch.tanh(emb.view(-1,block_size*embed_size) @ W1 + b1)
    logits = H1 @ O + o

    loss = F.cross_entropy(logits, Ytr[ix])




    for p in parameters:
        p.grad = None

    loss.backward()

    with torch.no_grad():
        for p in parameters:
            p.data += -lr * p.grad




    if epoch % 100 == 0:
        print(f'{loss=:.6f}, {epoch=}, {lr=}')

    if epoch > epochs*0.8 and lr > 0.01:
        print('Adjusting lr')
        lr = 0.01
        batch_size = batch_size*2



with torch.no_grad():
    emb = C[Xdev]
    H1 = torch.tanh(emb.view(-1,block_size*embed_size) @ W1 + b1)
    logits = H1 @ O + o
    loss = F.cross_entropy(logits, Ydev)
    print(f'{loss.item():.6f} - Full loss', )

exit()


lr = 0.5

for i in range(10000):
    logits1 = (xenc.view(len(Y), -1) @ W) + b
    logits11 = torch.tanh(logits1)
    logits2 = logits11 @ W2 + b2
    loss = F.cross_entropy(logits2, Y)


    W.grad = None 
    b.grad = None
    W2.grad = None
    b2.grad = None
    loss.backward()


    with torch.no_grad():
        W += -lr * W.grad
        b += -lr * b.grad
        W2 += -lr * W2.grad
        b2 += -lr * b2.grad

    if i % 10 == 0:
        print(i, f'{loss.item():.6f}')
print(f'Final loss {loss.item():.6f}')











def sample(num=1):
    for i in range(num):
        word = ""
        context = torch.tensor([[0] * block_size])
        while True:
            cenc = F.one_hot(context,num_classes=27).float().view(1,-1)
            clog1 = (cenc @ W + b).view(-1)
            clog11 = torch.tanh(clog1)
            clog2 = clog11 @ W2 + b2



            probs = F.softmax(clog2, dim=0)

            ix = torch.multinomial(probs, 1, replacement=True).item()
            if ix == 0:
                break

            new_element = torch.tensor([[ix]])  
            context = torch.cat((context[:, 1:], new_element), dim=1)


            word += itos[ix]
        print(word)

sample(20)

exit()

# 2.3457, 1000 epocts
# Statistical trigram 2.185652256011963



end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.6f} seconds")
