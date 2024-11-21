from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np
import math

def visualize_training(frames,speed):
    fig, ax = plt.subplots()

    x_values = [x for x in np.arange(0,20,0.25)]
    y_values =  [math.sin(x) for x in x_values]
    line1, = ax.plot(x_values, y_values)

    def update(frame):
        line1.set_data(x_values[:frame], y_values[:frame])

        #ax.relim()       # Recalculate limits based on data
        #ax.autoscale_view()
        ax.set_xlim(0,10)
        ax.set_ylim(-1, 1)

    ani = FuncAnimation(fig, update, frames=frames,interval=speed,repeat=True)
    print('Kom hy hier')
    plt.show()

visualize_training(50,100)

