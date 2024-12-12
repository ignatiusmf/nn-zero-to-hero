import os
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
    global start_time
    print('-'*50, f'\n{(time.time() - start_time):.3f} - Elapsed time {note}\n', '-'*50)
    start_time = time.time()

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    print("CUDA is not available. Using CPU tensors.")

def mst(tensor, title=None, **kwargs):
    ## TENSOR VISUALIZATION ##
    plt.matshow(tensor.detach().cpu().numpy(), **kwargs)
    plt.title(title)
    plt.show()

def ht(tensor, title=None, **kwargs):
    ## HISTOGRAM ## 
    plt.hist(tensor.view(-1).tolist(),50)
    plt.title(title)
    plt.show()

def ht2(tensor1, tensor2, title1=None, title2=None, **kwargs):
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.hist(tensor1.view(-1).tolist(),50, density=True)
    plt.title(title1)
    plt.subplot(122)
    plt.hist(tensor2.view(-1).tolist(),50, density=True)
    plt.title(title2)
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
###########################################################################3

words = open(os.path.dirname(os.path.abspath(__file__)) + '/names.txt', 'r').read().splitlines()
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
n_hidden = 100
vocab_size = len(chars) + 1

C = torch.randn((vocab_size,embed_size)) 
W1 = torch.randn((block_size*embed_size,n_hidden)) * 5/3/(block_size*embed_size)**0.5
#b1 = torch.randn((n_hidden)) * 0.01
O = torch.randn((n_hidden,vocab_size)) * 0.01
o = torch.randn((vocab_size)) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, O, o, bngain, bnbias]

for p in parameters:
    p.requires_grad = True


timer('before training')

lr = 0.1
batch_size = 32 

epochs = 10000
lossi = []
for epoch in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    emb = C[Xtr[ix]]
    preH1 = emb.view(-1,block_size*embed_size) @ W1 #+ b1

    bnmeani = preH1.mean(0, keepdim=True)
    bnstdi = preH1.std(0, keepdim=True)

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    preH1 = bngain * (preH1 - bnmeani) / bnstdi  + bnbias

    H1 = torch.tanh(preH1)
    logits = H1 @ O + o
    loss = F.cross_entropy(logits, Ytr[ix])

    lossi.append(loss.log10().item())

    for p in parameters:
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in parameters:
            p.data += -lr * p.grad


    if epoch % 100 == 0:
        print(f'{loss=:.6f}, {epoch=}, {lr=}')

    if epoch > epochs*0.8 and lr > 0.01:
        print('\nAdjusting learning rate and batch size')
        lr = 0.01
        batch_size = batch_size*2

timer('after training')

def dev_eval():
    with torch.no_grad():
        emb = C[Xdev]
        preH1 = emb.view(-1,block_size*embed_size) @ W1 #+ b1 
        preH1 = bngain * (preH1 - bnmean_running) / bnstd_running  + bnbias
        H1 = torch.tanh(preH1)
        logits = H1 @ O + o
        loss = F.cross_entropy(logits, Ydev)
        print(f'{loss.item():.6f} - Dev loss', )
dev_eval()



plt.plot(np.convolve(lossi, np.ones(50)/50, mode='valid'))
plt.show()



plt.subplot(121)
plt.hist(H1.view(-1).tolist(), 50, density=True)
plt.title('H1')
plt.subplot(122)
plt.hist(preH1.view(-1).tolist(), 50, density=True)
plt.title('H1 Preact')
plt.show()


plt.imshow(H1.abs().detach().cpu().numpy() > 0.99)
plt.show()









# 2.3076 voor batch norm
# 2.361
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



