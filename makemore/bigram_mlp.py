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

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    print("CUDA is not available. Using CPU tensors.")

###########################################################################3

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  
itos = {i:s for s,i in stoi.items()} 




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


xenc = F.one_hot(xs, num_classes=27).float() 
W = torch.randn((27,27), requires_grad=True)


logits = xenc @ W
print(logits[1])
print(logits.shape)



exit()

epochs = 100 
lastloss = 0
for epoch in range(epochs):
    logits = xenc @ W
    loss = F.cross_entropy(logits, ys)

    W.grad = None 
    loss.backward()

    with torch.no_grad():
        W += -15 * W.grad

    if (epoch + 1) % (epochs / 100) == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Loss Delta: {(loss - lastloss):.6f}")
        lastloss = loss




iterations = 10
prob_w = torch.exp(W) 
prob_w = prob_w / prob_w.sum(dim=1, keepdim=True)
names = []
for i in range(iterations):
    name = ""
    letter_index = 0
    while(True):
        letter_index = torch.multinomial(prob_w[letter_index], 1, replacement=True).item()
        if letter_index == 0:
            break
        name += itos[letter_index]
    names.append(name)
    print(name)





end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.6f} seconds")







