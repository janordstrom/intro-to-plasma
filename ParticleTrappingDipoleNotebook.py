# -*- coding: utf-8 -*-
"""
Particle trapping in a dipole magnetic field
Julia Nordstrom
Spring 2026
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

import astropy.units as u
from plasmapy.particles import Particle
from plasmapy.plasma.grids import CartesianGrid
from plasmapy.simulation.particle_tracker.particle_tracker import ParticleTracker
from plasmapy.formulary.magnetostatics import MagneticDipole
from plasmapy.simulation.particle_tracker.termination_conditions import TimeElapsedTerminationCondition
from plasmapy.simulation.particle_tracker.save_routines import IntervalSaveRoutine

# input parameters
#moment=8e22*u.amp/u.m/u.m # magnetic moment, approx strength of earth's magnetic moment
moment = 1*u.amp*u.m*u.m
o = (0,0,0)*u.m # origin

gridlength=20
gridstart =-10 
gridend=10
grid = CartesianGrid(gridstart*u.m,gridend*u.m,num=gridlength)
xx=np.linspace(gridstart,gridend,gridlength)
yy=np.linspace(gridstart,gridend,gridlength)
zz=np.linspace(gridstart,gridend,gridlength)

dipole = MagneticDipole(moment,o) # initialize dipole
Bx = np.zeros(grid.shape)*u.T/u.m/u.m/u.m #initialize B array
By = np.zeros(grid.shape)*u.T/u.m/u.m/u.m
Bz = np.zeros(grid.shape)*u.T/u.m/u.m/u.m


for i in np.arange(gridlength):
    for j in np.arange(gridlength):
        for k in np.arange(gridlength):
            point =(grid.pts0.value[i,j,k],grid.pts1.value[i,j,k],grid.pts2.value[i,j,k])*u.m
            B_calc = dipole.magnetic_field(point) # create dipole field
            Bx[i,j,k] = B_calc[0]
            By[i,j,k] = B_calc[1]
            Bz[i,j,k] = B_calc[2]
            
            
#%%
grid.add_quantities(B_x=Bx*u.m*u.m*u.m,B_y=By*u.m*u.m*u.m,B_z=Bz*u.m*u.m*u.m)
termination_condition = TimeElapsedTerminationCondition(25*u.s)
save_routine = IntervalSaveRoutine(.001*u.s)
simulation = ParticleTracker(
    grid, save_routine=save_routine,
    termination_condition=termination_condition,
    verbose=False,
)


x0=[[.5, .5, .5]]*u.m
v0=[[.1,0,0]]*u.m/u.s
particle = Particle("e-")
simulation.load_particles(x0, v0, particle)
simulation.run()

results = save_routine.results
particle_trajectory = results["x"][:, 0]
particle_position_x = particle_trajectory[:, 0]
particle_position_y = particle_trajectory[:, 1]
particle_position_z = particle_trajectory[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(particle_position_x, particle_position_y,particle_position_z)
#%%
plt.figure()
plt.contour(xx,zz,Bz[:,gridlength//2,:].value) # side view of dipole
plt.xlim(-.25,.25)
plt.ylim(-.25,.25)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Bz')

plt.figure()
plt.contour(xx,yy,Bz[:,:,gridlength//2].value) # top down view of dipole
plt.xlim(-.25,.25)
plt.ylim(-.25,.25)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bz')
#%%
plt.figure()
plt.streamplot(xx,zz,Bx[:,5,:].value,Bz[:,5,:].value)
plt.figure()
plt.streamplot(yy,zz,By[5,:,:].value,Bz[5,:,:].value)
plt.figure()
plt.streamplot(xx,yy,Bx[:,:,5].value,By[:,:,5].value)
#%%
fig = go.Figure(data=go.Streamtube(x=xx,y=yy,z=zz,u=Bz[:,20,20].value,v=Bz[20,:,20].value,w=Bz[20,20,:].value))
fig.show()
#%%
termination_condition = TimeElapsedTerminationCondition(25*u.s)

simulation = ParticleTracker(
    grid,
    save_routine=save_routine,
    termination_condition=termination_condition,
    verbose=False,
)

#%%
dipole = MagneticDipole(
    np.array([0, 0, 1]) * u.A * u.m * u.m, np.array([0, 0, 0]) * u.m
)
print(dipole)
