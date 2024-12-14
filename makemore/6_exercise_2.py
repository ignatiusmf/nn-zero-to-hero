import random
import torch
import os

random.seed(0)
torch.manual_seed(0)
torch.set_default_device('cuda')

words = open(os.path.dirname(os.path.abspath(__file__)) + '/names.txt').read().split()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 2


def build_dataset(words):
    X, Y = [], []
    for w in words[:3]:
        context = [0] * block_size
        for c in w + '.':
            X.append(context)
            Y.append(stoi[c])
            # print(''.join(itos[con] for con in context), '-->', c )
            context = context[1:] + [stoi[c]]

    X, Y = torch.tensor(X), torch.tensor(X)
    


    return X, Y 






n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
# random.shuffle(words)

Xtr, Ytr = build_dataset(words[:n1])
# Xde, YDe = build_dataset(words[n1:n2])
# Xte, Yte = build_dataset(words[n2:])



