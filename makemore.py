import warnings
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(2)


def hypothesis1():
# NOTE: HYPOTHESIS: P1[:, 1] = P2[1]. 
# HYPOTHESIS: Plausible. For some reason werk dit nie uit nie. Maar N1[:, 1] = N2[1]

    P1, N1 = create_trigram_probability_distribution()
    P2, N2 = create_bigram_probability_distribution()

    test = torch.zeros((27), dtype=torch.float)
    for i in range(27):
        test += P1[i,1]
    test = test.float() / test.float().sum()
    print(test)

    print()

    counts = torch.zeros((27), dtype=torch.int32)
    for i in range(27):
        counts += N1[i, 1]
    pprint(N2[1])
    pprint(counts)

    count_prob = counts.float() / counts.float().sum()
    pprint(count_prob.sum())

    pprint("\n retry")

    retry = torch.zeros((27), dtype=torch.float)
    for i in range(27):
        retry += P1[:, 1, i]
    print(retry.float().sum())
    pprint(retry)


def hypothesis2():
# NOTE: HYPOTHESIS: Die manually created probability distribution moet dieselfde wees as n synthetic een wat gemaak is deur n large sample van woorde generate deur die eerste distriubtion. 
# NOTE: HYPOTHESIS: Confirmed, die synthetic een approach die ground truth een
    names = generate_name(100000)
    P1, N1 = create_bigram_probability_distribution()
    CP1, CN1 = create_bigram_probability_distribution(names)

    print(P1[0])
    print('\n')
    print(CP1[0])

    compare_distributions(CP1, P1)


def hypothesis3():
# NOTE: Hypothesis: TRIGRAM/BIGRAM probability distrubtion wat generated is van TRIGRAM/BIGRAM generated words moet moet die TRIGRAM/BIGRAM probability distibution generated van die words dataset approach.
# NOTE: Hypothesis: Confirmed 
    tri_names = generate_name_trigram(100000)
    bi_names = generate_name(100000)


    distros = {
        "B_GT": create_bigram_probability_distribution(),
        "B_B": create_bigram_probability_distribution(bi_names),
        "B_T": create_bigram_probability_distribution(tri_names),
        "T_GT": create_trigram_probability_distribution(),
        "T_B": create_trigram_probability_distribution(bi_names),
        "T_T": create_trigram_probability_distribution(tri_names) 
    }

    for k,v in distros.items():
        print('\n', k)
        for k1,v1 in distros.items():
            if k == k1:
                print('skipping', k, k1)
                continue

            print(k,k1)
            if k[0] == "B" and k1[0] == "T":
                p1, p2 = v[0], v1[1]
            elif k[0] == "T" and k1[0] == "B":
                p1, p2 = v[1], v1[0]
            elif k[0] == "T" and k1[0] == "T":
                p1, p2 = v[1], v1[1]
            else:
                p1,p2 = v[0], v1[0]

            compare_distributions(p1, p2)



def flatten_tensor(counts): # TODO: THIS SHIT AINT WORKING. HOEKOM KAN EK NIE DIE PROBABILITIES VAN MY TRIGRAM TENSOR TRANSFORM IN DIESELFDES VAN MY BIGRAM TENSOR??? MAAR COUNTS WERK!?
    flattened_counts = torch.zeros((27,27), dtype=torch.int32)
    for i in range(27):
        for j in range(27):
            flattened_counts[i] += counts[j, i]

    flattened_probs = torch.zeros((27,27), dtype=torch.float)
    for i in range(27):
        flattened_probs[i] = flattened_counts[i].float() / flattened_counts[i].float().sum()
    return flattened_probs, flattened_counts

def compare_distributions(dist1, dist2):
    if len(dist1.shape) == 3 and len(dist2.shape) == 2:
        dist1,counts = flatten_tensor(dist1)
    elif len(dist2.shape) == 3 and len(dist1.shape) == 2:
        dist2,counts = flatten_tensor(dist2)
    elif len(dist1.shape) == 3 and len(dist2.shape) == 3:
        dist1, counts = flatten_tensor(dist1)
        dist2, counts = flatten_tensor(dist2) # TODO: THIS SHIT AINT WORKING. HOEKOM KAN EK NIE TRIGRAM MODELS MET MEKAAR COMPARE NIE???
        #dist1 = dist1.view(-1, dist1.size(-1))  
        #dist2 = dist2.view(-1, dist2.size(-1)) 

    epsilon = 1e-10  
    kl_divergence = F.kl_div((dist1 + epsilon).log(), dist2 + epsilon, reduction='batchmean')
    print("KL Divergence:", kl_divergence.item())

def plot_count_matrix():
    plt.figure(figsize=(100,100), dpi=300)
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize="2")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray", fontsize="2")
    plt.axis("off")





def initialize_data_create_lookup_tables(custom_list_of_words=False):
    if custom_list_of_words:
        words = custom_list_of_words
    else:
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

    print("Custom words evaluated by bigram model", evaluate_model(P, ["ignatio", "kyle", "suzaan", "luane", "albert", "amelie"]))
    print("Custom words evaluated by pentgram model", evaluate_model(P3, ["ignatio", "kyle", "suzaan", "luane", "albert", "amelie"]))



# P, N = create_bigram_probability_distribution()
# print("Normal words evaluated by bigram model", evaluate_model(P))

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




## TODO: Add nog n layer op hierdie neural network. Clean up the code maybe.



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
