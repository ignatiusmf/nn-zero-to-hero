import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from pprint import pprint
import torch

## TODO: Implement Batching when trianing. Dink dit sal help vir die slegte results in die SIN grafiek approximation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),        
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


def train(epoch, epochs, batch_size):
    if shuffle:
        permutation = torch.randperm(x_tensor.size(0))
        x_shuffled = x_tensor[permutation]
        y_shuffled = y_tensor[permutation]
    else:
        x_shuffled = x_tensor
        y_shuffled = y_tensor

    for i in range(0, x_tensor.size(0), batch_size):
        batch_x = x_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]

        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    if (epoch + 1) % (epochs / 10) == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



# NOTE: Hoekom train die neural network altyd eers om x = 0 ? Dit gebeur selfs wanneer ek die training data by negatief of 0 begin. En as ek hom by, bv, 5, begin, dan leer hy die begin van die graviek eers
# NOTE: En hy leer dit baie stadiger as ek hom by 5 begin ipv 0






# TODO: maar ver af, ek wil n graph generate wat die evaluation loss wys, dan ken mens dit vergelyk met die training loss.


from matplotlib.animation import FuncAnimation

def train_and_visualize_loss(train_steps,speed, batch_size):
    fig, ax = plt.subplots()
    steps = np.arange(train_steps) 
    losses = []

    line1, = ax.plot([], [])

    def update(frame):
        train(frame, train_steps, batch_size)

        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        losses.append(loss.item())

        line1.set_data(steps[:frame],np.sqrt(losses[:frame]))
        ax.relim()       # Recalculate limits based on data
        ax.autoscale_view()

    ani = FuncAnimation(fig, update, frames=train_steps,interval=speed,repeat=False)
    plt.show()


def visualize_training(train_steps,speed,batch_size,trange):
    fig, ax = plt.subplots()


    x_v = np.arange(trange[0], trange[1], 0.01)
    y_v = np.sin(x_v)
    x_v_t = torch.tensor(x_v, dtype=torch.float32).view(-1,1)
    y_pred = model(x_v_t).detach().numpy()

    line1, = ax.plot(x_v,y_v)
    line2, = ax.plot([], [])

    def update(frame):
        train(frame, train_steps, batch_size)

        y_pred = model(x_v_t).detach().numpy()
        
        line2.set_data(x_v_t, y_pred)
        ax.relim()       
        ax.autoscale_view()
        #ax.set_ylim(-5,5)

    ani = FuncAnimation(fig, update, frames=train_steps,interval=speed,repeat=False)
    plt.show()

# Visualize graph
def vis_graph(x_start,x_end):
    x = np.arange(x_start, x_end, 0.01)
    x_t = torch.tensor(x, dtype=torch.float32).view(-1,1)
    y = model(x_t).detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_ylim(-10000,10000)
    ax.set_xlim(-10000,10000)
    plt.show()







epochs = 1000
speed = 0
shuffle = True 
learning_rate = 0.001
batch_amount = 15

training_range = (-50, 50, 0.01)


seed = 1 
torch.manual_seed(seed)
np.random.seed(seed)

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

x = np.arange(training_range[0], training_range[1], training_range[2])
y = np.sin(x)
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

batch_size = round(x_tensor.size(0) / batch_amount)

train_and_visualize_loss(epochs, speed, batch_size)
visualize_training(epochs, speed, batch_size,training_range) 
#visualize_training(epochs, speed, batch_size, (-1000,1000))
vis_graph(-10000, 10000)











# setup
def vis_weights(train_steps, speed, batch_size):
    axrange = (-5,5)
    fig, ax = plt.subplots()
    ax.set_xlim(axrange[0], axrange[1])
    ax.set_ylim(axrange[0], axrange[1])
    plt.axhline(0, color='black', linewidth=1.5)  
    plt.axvline(0, color='black', linewidth=1.5)


    x_t = np.arange(axrange[0], axrange[1], 0.1)
    x_tt = torch.tensor(x_t, dtype=torch.float32).view(-1,1)

    weights = model.layers[-1].weight.data
    bias = model.layers[-1].bias.data
    xws = x_tt @ weights 

    lines = [ax.plot([], [])[0] for _ in range(xws.shape[1])]  # Create 10 empty line objects

    def update(frame):
        train(frame, train_steps, batch_size)

        weights = model.layers[-1].weight.data
        bias = model.layers[-1].bias.data
        xws = x_tt @ weights 

        for i, line in enumerate(lines):
            line.set_data(x_t, xws[:,i] + bias)  
        return lines

    ani = FuncAnimation(fig, update, frames=train_steps, interval=speed, blit=True, repeat=False)

    plt.show()




vis_weights(epochs, speed, batch_size) 
