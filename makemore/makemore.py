import warnings
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(2)

def initialize_data_create_lookup_tables():
    words = open('names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0  
    itos = {i:s for s,i in stoi.items()} 
    return stoi, itos, words

stoi, itos, words = initialize_data_create_lookup_tables()

xs, ys = [], [] 
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print('ys', len(ys))

seed = 766183
torch.manual_seed(seed)


xenc = F.one_hot(xs, num_classes=27).float() 
W = torch.randn((27,27), requires_grad=True)

def train(epoch,epochs,neurons):
    W = neurons
    logits = xenc @ W
    counts = torch.exp(logits)
    probs = counts / counts.sum(dim=1, keepdim=True)

    loss = -probs[torch.arange(probs.size(0)), ys].log().mean()

    W.grad = None 
    loss.backward()

    with torch.no_grad():
        W += -50 * W.grad

    if (epoch + 1) % (epochs / 100) == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


epochs = 100
for epoch in range(epochs):
    train(epoch,epochs,W)

plt.matshow(W.detach().numpy())
plt.show()



def sample(iterations,W):
    # P, N = create_bigram_probability_distribution()

    prob_w = torch.exp(W) 
    prob_w = prob_w / prob_w.sum(dim=1, keepdim=True)
    names = []
    for i in range(iterations):
        name = ""
        letter_index = 0
        while(True):
            letter_index = torch.multinomial(prob_w[letter_index], 1, replacement=True, generator=g).item()
            if letter_index == 0:
                break
            name += itos[letter_index]
        names.append(name)
        print(name)
    return names

sample(10,W)
