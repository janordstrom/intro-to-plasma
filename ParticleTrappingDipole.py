# -*- coding: utf-8 -*-
"""
Particle trapping in a dipole magnetic field
Julia Nordstrom
Spring 2026
"""

# imports
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from plasmapy.particles import Particle
from plasmapy.plasma.grids import CartesianGrid
from plasmapy.simulation.particle_tracker.particle_tracker import ParticleTracker
from plasmapy.formulary.magnetostatics import MagneticDipole

# input parameters
#moment=8e22*u.amp/u.m/u.m # magnetic moment, approx strength of earth's magnetic moment
moment = 1*u.amp*u.m*u.m
o = (0,0,0)*u.m # origin

gridlength=20
grid = CartesianGrid(-1*u.m,1*u.m,num=gridlength)

dipole = MagneticDipole(moment,o) # initialize dipole
B = np.zeros(grid.shape)*u.T/u.m/u.m/u.m #initialize B array

for i in np.arange(gridlength):
    for j in np.arange(gridlength):
        for k in np.arange(gridlength):
            point =(grid.pts0.value[i,j,k],grid.pts1.value[i,j,k],grid.pts2.value[i,j,k])*u.m
            point=(-1,-1,-1)*u.m
            B[i,j,k] = dipole.magnetic_field(point) # create dipole field

