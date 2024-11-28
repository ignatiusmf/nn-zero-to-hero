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

def mst(tensor, title=None, **kwargs):
    plt.matshow(tensor.detach().cpu().numpy(), **kwargs)
    plt.title(title)
    plt.show()

def ht(tensor, title=None, **kwargs):
    plt.hist(tensor.view(-1).tolist(),50)
    plt.title(title)
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
W1 = torch.randn((block_size*embed_size,100)) * 0.1
b1 = torch.randn((100)) * 0.1
O = torch.randn((100,27)) * 0.01
o = torch.randn((27)) * 0

parameters = [C, W1, b1, O, o]

for p in parameters:
    p.requires_grad = True


lr = 0.1
batch_size = 32 

epochs = 5000
lossi = []
for epoch in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    emb = C[Xtr[ix]]
    preH1 = emb.view(-1,block_size*embed_size) @ W1 + b1
    H1 = torch.tanh(preH1)
    logits = H1 @ O + o
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None

    loss.backward()
    lossi.append(loss.log10().item())

    with torch.no_grad():
        for p in parameters:
            p.data += -lr * p.grad


    if epoch % 100 == 0:
        print(f'{loss=:.6f}, {epoch=}, {lr=}')

    if epoch > epochs*0.8 and lr > 0.01:
        print('Adjusting learning rate and batch size')
        lr = 0.01
        batch_size = batch_size*2




plt.plot(lossi)
plt.show()


mst(preH1, 'h1preact')
ht(preH1, 'h1preact')

mst(H1, 'h1')
ht(H1, 'h1')


mst(logits)
ht(logits)


with torch.no_grad():
    emb = C[Xdev]
    H1 = torch.tanh(emb.view(-1,block_size*embed_size) @ W1 + b1)
    logits = H1 @ O + o
    loss = F.cross_entropy(logits, Ydev)
    print(f'{loss.item():.6f} - Dev loss', )

exit()































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
