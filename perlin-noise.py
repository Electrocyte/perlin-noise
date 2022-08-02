#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:16:12 2022

@author: mangi
"""

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot(x,xx):
    f, ax = plt.subplots(figsize=(15, 15))
    sns.lineplot(x=x, y=xx)
    ax.set_xlabel("FREQUENCY", size=40)
    ax.set_ylabel("amplitude", size=40)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    plt.show()
    
# amplitude = y
# persistence controls amplitude
# FREQUENCY = x
# FREQUENCY might actually be more accurate here
# lacunarity controls time

def f(x):
    return np.sin(x) + np.random.normal(scale=0.1, size=len(x))

FREQUENCY = np.linspace(1, 15, 200) # frequency values
# simple random noise, quite jagged
amplitude = f(FREQUENCY)

# plot(FREQUENCY, amplitude)

amplitude1 = np.sin(FREQUENCY)

cos_amp = np.cos(FREQUENCY)

plot(FREQUENCY, amplitude1)
plot(FREQUENCY, cos_amp)

def power_vals(x: list):
    vals = [2**val for val in x]
    return vals

def power_inv_vals(x: list):
    vals = [(1/val**2) for val in x]
    return vals

powers = [1,2,3]
lacunarity = power_vals(powers)

xs = [x*lacunarity[1] for x in FREQUENCY]
amplitude2 = f(xs)
xxs = [x*lacunarity[2] for x in FREQUENCY]
amplitude3 = f(xxs)

persistence = power_inv_vals(powers)
persistence2 = [x*persistence[1] for x in amplitude2]
persistence3 = [x*persistence[2] for x in amplitude3]
persistence4 = [x*persistence[2] for x in cos_amp]
persistence5 = [x*persistence[1] for x in cos_amp]

# plot(time, amplitude1)
# plot(xxs, persistence3)

from operator import add
nn = list( map(add, persistence4, persistence5) )
nnn = list( map(add, nn, amplitude1) )
noises = list( map(add, persistence2, nnn) )
noisier = list( map(add, noises, persistence3) )

# time currently unhinged from amplitude...
plot(FREQUENCY, noisier)

# https://stackoverflow.com/questions/60350598/perlin-noise-in-pythons-noise-library
import noise
import numpy as np
# from PIL import Image

shape = (1024,1024)
scale = .5
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = np.random.randint(0,100)

world = np.zeros(shape)

# make coordinate grid on [0,1]^2
x_idx = np.linspace(0, 1, shape[0])
y_idx = np.linspace(0, 1, shape[1])
world_x, world_y = np.meshgrid(x_idx, y_idx)

# apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
# world = np.vectorize(noise.pnoise2)(world_x/scale,
#                         world_y/scale,
#                         octaves=octaves,
#                         persistence=persistence,
#                         lacunarity=lacunarity,
#                         repeatx=1024,
#                         repeaty=1024,
#                         base=seed)

# # here was the error: one needs to normalize the image first. Could be done without copying the array, though
# img = np.floor((world + .5) * 255).astype(np.uint8) # <- Normalize world first
# Image.fromarray(img, mode='L').show()