#!/usr/bin/env python
# coding: utf-8

# #  1D Diffusion Model

# https://github.com/csdms/roadshow-diffusion

# Here we develop a one-dimensional model of diffusion.
# It assumes a constant diffusivity.
# It uses a regular grid.
# It has a step function for an initial condition.
# It has fixed boundary conditions.

# Here is the diffusion equation:

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$

# Here is the discretized version of the diffusion equation we will solve with our model

# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

# This is the FTCS scheme as described by Slingerland and Kump (2011)

# We'll use two libraries , NumPy (for array) & Matplotlib (for plotting, that aren't a part of the core Python distribution.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# Start by setting two fixed model parameters, the diffusivity and size of the model domain.

# In[ ]:


D = 100
Lx = 300


# Next, set up the model grid using a NumPy array.

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)


# Set the initial conditions for the model.
# The cake 'C" is a step function with a high value of the left, a low value on the right, and a step at the center of the domain.

# In[ ]:


C = np.zeros_like(x)
c_left = 500
c_right = 0
C[x <= (Lx / 2)] = c_left # we could easily do this w/ subscripting instead of this
C[x > (Lx / 2)] = c_right


# Plot the initial profile

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial Profile")


# Set the number of time steps in the model. Calculate a stable timestep using a stability criterion.

# In[ ]:


nt = 5000
dt = 0.5 * dx ** 2 / D # "vo noiman" stability criterion


# Loop over the time step of the model, solving the diffusion equation using the FTCS scheme shown above.
# Note the use of  array operations on the variable 'C'.
# The boundary conditions remian fixed in each timestep.

# In[ ]:


for t in range(0, nt):
    C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])
# C[1:-1] because we are miantaining our boundary conditions 'c_left' & 'c_right'


# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial Profile")

