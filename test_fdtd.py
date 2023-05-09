import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,fftfreq,fftshift 

import fdtd


fd = fdtd.FDTD_Maxwell_1D()
            
x0 = 2.0; s0 = 0.25
initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
frec = fftshift(fftfreq(len(initialField), fd.dx))

FFTinitialField = fftshift(fft(initialField))

fd.a_p[fd.x>4] = -1 - 1j
fd.c_p[fd.x>4] = 1 + 0j
fd.a_p[fd.x>6] = 0 + 0j
fd.c_p[fd.x>6] = 0 + 0j


fd.eps_inf = 1.0
fd.eps_0 = 1.0

a_p_int = fd.a_p[50]
c_p_int = fd.c_p[50]

frec_max = 20
Nfrec = 100
frecfd = np.linspace(0, frec_max, num = Nfrec)

Epsilon = np.zeros(frecfd.shape, dtype=np.complex_)
for w in range(0, len(frecfd)):
    Epsilon[w] = fd.eps_0 * fd.eps_inf + fd.eps_0 * c_p_int / (frecfd[w] * 1j - a_p_int) + fd.eps_0 * np.conj(c_p_int) / (frecfd[w] * 1j -  np.conj(a_p_int)) 

t_max = 10

fd.e[:] = initialField[:]
e_td = np.zeros(int(t_max/fd.dt))
e_rd = np.zeros(int(t_max/fd.dt))

for _ in np.arange(0, t_max, fd.dt):
    fd.stepMod()
    e_td[int(_/fd.dt)] = fd.e[70]
    e_rd[int(_/fd.dt)] = fd.e[30]

    
    # plt.plot(fd.x, fd.e, '*')
    # plt.plot(fd.xDual, fd.h, '.')
    # plt.plot(fd.x, np.abs(fd.J), '.')
    # plt.ylim(-1.1, 1.1)
    # plt.xlim(fd.x[0], fd.x[-1])
    # plt.grid()
    # plt.pause(0.01)
    # plt.cla()
    

fn = fdtd.FDTD_Maxwell_1D()

fn.e[:] = initialField[:]
fn.eps_inf = 1.0
fn.eps_0 = 1.0

e_t = np.zeros(int(t_max/fd.dt))


for _ in np.arange(0, t_max, fd.dt):
    fn.step()
    e_t[int(_/fn.dt)] = fn.e[30]
       
    R = np.corrcoef(initialField, fn.e)

    #plt.plot(fn.x, fn.e, '*')
    #plt.plot(fn.xDual, fn.h, '.')
    #plt.plot(fn.x, np.abs(fn.J), '.')
    #plt.ylim(-1.1, 1.1)
    #plt.xlim(fn.x[0], fn.x[-1])
    #plt.grid()
    #plt.pause(0.01)
    #plt.cla()

#plt.figure()
#plt.plot(frecfd, np.real(Epsilon))
#plt.plot(frecfd, np.imag(Epsilon))
#plt.show()

N_t = t_max / fd.dt
t_range = np.linspace(0, t_max, num = int(N_t))
    
# plt.figure()
# plt.plot(t_range, e_t)
# plt.plot(t_range, e_td)
# plt.plot(t_range, e_rd - e_t)
# plt.show()    
# plt.cla()

frecT = fftshift(fftfreq(len(t_range), fd.dt))

FFT_si= fftshift(fft(e_t))
FFT_sr= fftshift(fft(e_rd - e_t))
FFT_st= fftshift(fft(e_td))

plt.figure()
plt.plot(frecT, FFT_si)
plt.plot(frecT, FFT_sr)
plt.plot(frecT, FFT_st)
plt.show()       
   