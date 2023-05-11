import numpy as np
import matplotlib.pyplot as plt

import fdtd

fd = fdtd.FDTD_Maxwell_1D()

fd.d = 0.3

max_freq = 20
n_freq = 100
freq = np.linspace(0, max_freq, num = n_freq)

#eps_c = np.zeros(freq.shape, dtype=np.complex_)
eps_c = [fd.theoretical_test(w) for w in freq]
# plt.plot(freq, np.real(eps_c))
# plt.plot(freq, np.imag(eps_c))
# plt.show()


gamma = [1j * freq[i] * np.sqrt(fd.mu_0 * eps_c[i]) for i in range(n_freq)]
eta = [np.sqrt(fd.mu_0 / eps_c[i]) for i in range(n_freq)]
t_matrices = [np.array([[np.cosh(gamma[i] * fd.d), eta[i] * np.sinh(gamma[i] * fd.d)],
                        [1 / eta[i] * np.sinh(gamma[i] * fd.d), np.cosh(gamma[i] * fd.d)]])
              for i in range(n_freq)]

# Transmitance
T = [2 * fd.eta_0 / (t[0][0] * fd.eta_0 + t[0][1] + t[1][0] * fd.eta_0 ** 2 + t[1][1] * fd.eta_0) for t in t_matrices]

# Reflectivity
R = [(t[0][0] * fd.eta_0 + t[0][1] - t[1][0] * fd.eta_0 ** 2 - t[1][1] * fd.eta_0) / (
            t[0][0] * fd.eta_0 + t[0][1] + t[1][0] * fd.eta_0 ** 2 + t[1][1] * fd.eta_0)
     for t in t_matrices]

T_R = [np.abs(T[i] + R[i]) for i in range(len(T))]

plt.plot(freq, np.abs(T))
plt.plot(freq, np.abs(R))
plt.plot(freq, np.abs(T)**2 + np.abs(R)**2)
plt.xlim([0, 10])
plt.show()

