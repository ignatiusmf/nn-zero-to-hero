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


def create_bigram_probability_distribution(custom_list_of_words=False):
    if custom_list_of_words:
        stoi, itos, words = initialize_data_create_lookup_tables(custom_list_of_words)
    else:
        stoi, itos, words = initialize_data_create_lookup_tables()

    N = torch.zeros((27,27), dtype=torch.int32)
    for w in words:
        chars = ['.'] + list(w) + ['.'] 
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    P = torch.empty((27,27),dtype=torch.float)
    for i in range(27):
        P[i] = N[i].float() / N[i].float().sum()
    P += 0.00001
    return P, N

def generate_name(iterations):
    stoi, itos, words = initialize_data_create_lookup_tables()
    P, N = create_bigram_probability_distribution()
    names = []
    for i in range(iterations):
        name = ""
        letter_index = 0
        while(True):
            letter_index = torch.multinomial(P[letter_index], 1, replacement=True, generator=g).item()
            if letter_index == 0:
                break
            name += itos[letter_index]
        names.append(name)
    return names


def create_pentgram_probability_distribution(): 
    stoi, itos, words = initialize_data_create_lookup_tables()

    N = torch.zeros((27,27,27,27,27), dtype=torch.int32)
    for w in words:
        chars =  ['.'] + ['.'] + ['.'] + ['.'] + list(w) + ['.'] 
        for ch1,ch2,ch3,ch4,ch5 in zip(chars,chars[1:],chars[2:],chars[3:],chars[4:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            ix4 = stoi[ch4]
            ix5 = stoi[ch5]
            N[ix1, ix2, ix3, ix4, ix5] += 1

    P = torch.empty((27,27,27,27,27), dtype=torch.float)
    for i in range(27):
        for j in range(27):
            for k in range(27):
                for l in range(27):
                    P[i,j,k,l] = (N[i,j,k,l].float() + 0.001) / (N[i,j,k,l].float() + 0.001).sum()
    return P, N

def generate_name_pentgram(iterations):
    stoi, itos, words = initialize_data_create_lookup_tables()
    P, N = create_pentgram_probability_distribution()
    names = []
    for i in range(iterations):
        name = ""
        ix1,ix2,ix3,ix4 = 0,0,0,0 
        while(True):
            ix1,ix2,ix3,ix4 = ix2,ix3,ix4, torch.multinomial(P[ix1, ix2, ix3,ix4], 1, replacement=True, generator=g).item()
            if ix4 == 0:
                break
            name += itos[ix4]
        names.append(name)
    return names


def evaluate_model(P, custom_words=False):
    stoi, itos, words = initialize_data_create_lookup_tables()
    if custom_words:
        words = custom_words 
    n = 0
    nnll = 0.0

    size = len(P.size())
    if size == 2:
        for w in words:
            chars = ['.'] + list(w) + ['.'] 
            for ch1, ch2 in zip(chars, chars[1:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]

                ll = torch.log(P[ix1,ix2])
                nnll += ll 
                n += 1

    elif size == 5:
        for w in words:
            chars =  ['.'] + ['.'] + ['.'] + ['.'] + list(w) + ['.'] 
            for ch1,ch2,ch3,ch4,ch5 in zip(chars,chars[1:],chars[2:],chars[3:],chars[4:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]
                ix4 = stoi[ch4]
                ix5 = stoi[ch5]

                ll = torch.log(P[ix1,ix2,ix3,ix4,ix5])
                nnll += ll 
                n += 1

    nnll = -nnll/n
    return nnll.item()

def compare_models():
    names1 = generate_name(10)
    names2 = generate_name_pentgram(10)

    P, N = create_bigram_probability_distribution()
    P3, N = create_pentgram_probability_distribution()

    print("Normal words evaluated by bigram model", evaluate_model(P))
    print("Normal words evaluated by pentgram model", evaluate_model(P3))

    print("bigram words evaluated by bigram model", evaluate_model(P,names1))
    print("bigram words evaluated by pentgram model", evaluate_model(P3,names1))

    print("Pentgram words evaluated by bigram model", evaluate_model(P,names2))
    print("Pentgram words evaluated by pentgram model", evaluate_model(P3,names2))

    print("Custom words evaluated by bigram model", evaluate_model(P, ["ignatio", "kyle", "suzaan", "luane", "albert"]))
    print("Custom words evaluated by pentgram model", evaluate_model(P3, ["ignatio", "kyle", "suzaan", "luane", "albert"]))

compare_models()
