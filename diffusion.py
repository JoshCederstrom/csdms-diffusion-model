"""A 1D Diffusion Model."""

import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_profile(x, concentration, color="r"):
    plt.figure()
    plt.plot(x, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")

def calculate_stable_time_step(dx, diffusivity):
    return 0.5 * dx**2 / diffusivity

def step_like(x, step_at=0):
    y = np.empty_like(x, dtype=float)

    y[:step_at] = 0.0
    y[step_at] = 0.5
    y[step_at + 1 :] = 1.0

    return y

def new_profile(x, form="step"):
    match form:
        case "step":
            return step_like(x, step_at=len(x) // 2)
        case _:
            raise ValueError(f"unknown profile type ({form!r})")

def run_diffusion_model():
    D = 100
    Lx = 300
    
    dx = 0.5
    x = np.arange(start=0, stop=Lx, step=dx)
    nx = len(x)
    
    C = new_profile(x=x, form="step")
    
    plot_profile(x, concentration=C)
    
    nt = 5000
    dt = calculate_stable_time_step(dx, D)
    
    for t in range(0, nt):
        C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])
    
    plot_profile(x, C, color="b")

    return(C)


if __name__ == "__main__":
    cake = run_diffusion_model()
    np.savetxt(sys.stdout, cake, fmt="%.6f")