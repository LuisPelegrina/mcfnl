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
fd.a_p[fd.x> 5-fd.d/2] = -1 - 1j
fd.c_p[fd.x> 5-fd.d/2] = 1 + 0j
fd.a_p[fd.x> 5+fd.d/2] = 0 + 0j
fd.c_p[fd.x> 5+fd.d/2] = 0 + 0j

#Choose the time interaval that will be simulated
t_max = 10

#Create an array to save Electric Field at a given point
e_td = np.zeros(int(t_max/fd.dt))
e_rd = np.zeros(int(t_max/fd.dt))

#Start numerical method with dispersive material
for _ in np.arange(0, t_max, fd.dt):
    #Update a step
    fd.stepMod()

    #Save E before the dispesive region
    e_td[int(_/fd.dt)] = fd.e[70]

    #Save E after the dispesive region
    e_rd[int(_/fd.dt)] = fd.e[40]
    
#Declare a FTDT_Maxwell object to handle parameters of the simulation for a case without dispersive material
fn = fdtd.FDTD_Maxwell_1D()

#Create the same Electric field as the other example, and set some medium parameters
fn.e[:] = initialField[:]
fn.eps_inf = 1.0
fn.eps_0 = 1.0

#Create an array to save Electric Field before the dispersive region without having it
e_t = np.zeros(int(t_max/fd.dt))
e_t2 = np.zeros(int(t_max/fd.dt))

#Start numerical method with no dispersive material
for _ in np.arange(0, t_max, fd.dt):
    #Update a step without the dispersive material
    fn.step()

    #Save E after the dispesive region
    e_t[int(_/fn.dt)] = fn.e[40]
    e_t2[int(_/fn.dt)] = fn.e[70]   

#Plot time evolution of the electric wave
N_t = t_max / fd.dt
t_range = np.linspace(0, t_max, num = int(N_t))
plt.figure()
plt.plot(t_range, e_t)
plt.plot(t_range, e_td)
plt.plot(t_range, e_rd - e_t)
plt.xlim([0, 10])
plt.grid()
plt.xlabel("t")
plt.legend(["Initial Wave", "Transmited Wave", "Reflected Wave"])
plt.show()    
plt.cla()

#Make the fourier transforms of the Electric field evolution with time
frecT = fftshift(fftfreq(len(t_range), fd.dt))
FFT_si= fftshift(fft(e_t))
FFT_sr= fftshift(fft(e_rd - e_t))
FFT_st= fftshift(fft(e_td))

#Calculate T and R 
T = abs(FFT_st / FFT_si)
R = abs(FFT_sr / FFT_si)

#Plot T and R
plt.figure()
plt.plot(frecT*2*np.pi, T, ".-b")
plt.plot(frecT*2*np.pi, R, ".-r")
plt.plot(frecT*2*np.pi, T**2 + R**2, ".-g")
plt.ylim([0, 1])
plt.xlim([0, 20])
plt.grid()
plt.xlabel("ω")
plt.legend(["Transmitancia", "Reflectividad", "$|T|^2 + |R|^2$"])
plt.show()
