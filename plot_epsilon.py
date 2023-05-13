import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,fftfreq,fftshift 

import fdtd

#Declare a FTDT_Maxwell object to handle parameters of the simulation
fd = fdtd.FDTD_Maxwell_1D()
            
#Select the poles of the dispersive material
fd.a_p_t  = 1 + 1j
fd.c_p_t = -1 + 0j

#Calculate epsilon dependance with the frequency in the range [0,frec_max]
frec_max = 20
Nfrec = 100
frecfd = np.linspace(0, frec_max, num = Nfrec)
Epsilon = np.zeros(frecfd.shape, dtype=np.complex_)
Epsilon = [fd.theoretical_test(w) for w in frecfd]

#Plot epsilon real and imaginary part
plt.figure()
plt.plot(frecfd, np.real(Epsilon), "-b")
plt.plot(frecfd, np.imag(Epsilon), "-r")
plt.grid()
plt.ylabel("ε(ω)")
plt.xlabel("ω")
plt.legend(["Parte Real", "Parte Imaginaria"])
plt.show()
