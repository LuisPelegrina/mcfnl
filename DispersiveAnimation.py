import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,fftfreq,fftshift 

import fdtd

#Declare a FTDT_Maxwell object to handle parameters of the simulation
fd = fdtd.FDTD_Maxwell_1D()
            

#Define an initial electric field with gaussian shape
x0 = 2.0; s0 = 0.2
initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
fd.e[:] = initialField[:]

#Change the lenght of the dispersive material, the shaope of its poles
# and locate it at the middle of the simulated region
fd.d = 1

fd.a_p[fd.x> 5-fd.d/2] = -1 -1j
fd.c_p[fd.x> 5-fd.d/2] = 1 + 0j
fd.a_p[fd.x> 5+fd.d/2] = 0 + 0j
fd.c_p[fd.x> 5+fd.d/2] = 0 + 0j


#Choose the time interaval that will be simulated
t_max = 10

#Start the animation
for _ in np.arange(0, t_max, fd.dt):
    #Update a step
    fd.stepMod()
    
    #Plot a frame of the animation
    plt.plot(fd.x, fd.e, '*')
    plt.plot(fd.xDual, fd.h, '.')
    plt.plot(fd.x, np.abs(fd.J), '.')
    plt.ylim(-1.1, 1.1)
    plt.xlim(fd.x[0], fd.x[-1])
    plt.xlabel("x")
    plt.legend(["E", "H", "J"])
    plt.grid()
    plt.pause(0.01)
    plt.cla()
    