"""A 1D Diffusion Model."""

import numpy as np
import matplotlib.pyplot as plt


def plot_profile(x, concentration, color="r"):
    plt.figure()
    plt.plot(x, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")
    
D = 100
Lx = 300

dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)

C = np.zeros_like(x)
c_left = 500
c_right = 0
C[x <= (Lx / 2)] = c_left 
C[x > (Lx / 2)] = c_right

plot_profile(x, C)

nt = 5000
dt = 0.5 * dx ** 2 / D 

for t in range(0, nt):
    C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])

plot_profile(x, C, color="b")