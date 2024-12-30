import random
import torch 
import os 

random.seed(1)
torch.manual_seed(1)
torch.set_default_device('cuda')

words = open(os.path.dirname(os.path))
