import numpy as np
import matplotlib.pyplot as plt

import fdtd

# Declare a FTDT_Maxwell object to handle parameters of the simulation
fd = fdtd.FDTD_Maxwell_1D()

# Change the length of the dispersive material
fd.d = 0.3

# Set up the range of frequencies
max_freq = 20
n_freq = 40
freq = np.linspace(0, max_freq, num = n_freq)

# Compute the theoretical eps_c for a set of frequencies
eps_c = [fd.theoretical_test(w) for w in freq]

# Compute gamma (material's complez propagation constant)
gamma = [1j * freq[i] * np.sqrt(fd.mu_0 * eps_c[i]) for i in range(n_freq)]
# Compute parameter eta (intrinsic impedance)
eta = [np.sqrt(fd.mu_0 / eps_c[i]) for i in range(n_freq)]
# Compute transmission matrix for each frequency
t_matrices = [np.array([[np.cosh(gamma[i] * fd.d), eta[i] * np.sinh(gamma[i] * fd.d)],
                        [1 / eta[i] * np.sinh(gamma[i] * fd.d), np.cosh(gamma[i] * fd.d)]])
              for i in range(n_freq)]

# Compute transmitance, T, for each frequency
T = [2 * fd.eta_0 / (t[0][0] * fd.eta_0 + t[0][1] + t[1][0] * fd.eta_0 ** 2 + t[1][1] * fd.eta_0) for t in t_matrices]

# Compute reflectivity, R, for each frequency
R = [(t[0][0] * fd.eta_0 + t[0][1] - t[1][0] * fd.eta_0 ** 2 - t[1][1] * fd.eta_0) /
     (t[0][0] * fd.eta_0 + t[0][1] + t[1][0] * fd.eta_0 ** 2 + t[1][1] * fd.eta_0)
     for t in t_matrices]

# Compute the sum of the modules of T and R
T_R = [np.abs(T[i] + R[i]) for i in range(len(T))]

# Plot T, R and the sum
plt.plot(freq, np.abs(T), ".-b")
plt.plot(freq, np.abs(R), ".-r")
plt.plot(freq, np.abs(T)**2 + np.abs(R)**2, ".-g")
plt.ylim([0, 1])
plt.xlim([0, 12.5])
plt.grid()
plt.xlabel("ω")
plt.legend(["Transmitancia", "Reflectividad", "$|T|^2 + |R|^2$"])
plt.show()

