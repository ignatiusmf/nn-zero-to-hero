import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
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
            nn.Linear(10, 1)   
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


def visualize_training(train_steps,speed, batch_size):
    fig, ax = plt.subplots()

    y_pred = model(x_tensor).detach().numpy()

    line2, = ax.plot(x,y)
    line1, = ax.plot([], [])

    def update(frame):
        train(frame, train_steps, batch_size)

        y_pred = model(x_tensor).detach().numpy()
        
        line1.set_data(x, y_pred)
        ax.relim()       
        ax.autoscale_view()
        ax.set_ylim(-1,1)

    ani = FuncAnimation(fig, update, frames=train_steps,interval=speed,repeat=False)
    plt.show()



epochs = 1000
speed = 0
shuffle = False 
learning_rate = 0.001
batch_amount = 15

training_range = (0, 7, 0.1)


seed = 42
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
visualize_training(epochs, speed, batch_size)
































#epochs = 1 
#train_normal(epochs)

#y_pred = model(x_tensor).detach().numpy()

#plt.plot(x, y, label='True sin(x)')
#plt.plot(x, y_pred, label='MLP Prediction')
#plt.legend()
#plt.show()
