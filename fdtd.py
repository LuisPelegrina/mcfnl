import numpy as np
import matplotlib.pyplot as plt

class FDTD_Maxwell_1D():
    def __init__(self, L=10, CFL=1.0, Nx=101):
        self.x = np.linspace(0, L, num=Nx)
        self.d = 0.4
        self.xDual = (self.x[1:] + self.x[:-1])/2
       
        self.eps_0 = 1.0
        self.eps_inf = 1.0
        self.mu_0 = 1.0
        self.eta_0 = np.sqrt(self.mu_0 / self.eps_0)
        self.c_0 = 1/np.sqrt(self.eps_0*self.mu_0)
        self.sigma = 0

        self.a_p = np.zeros(self.x.shape, dtype=np.complex_)
        self.c_p = np.zeros(self.x.shape, dtype=np.complex_)

        self.a_p_t = -1 + -1j
        self.c_p_t = 1 + 0j

        self.dx = self.x[1] - self.x[0]
        self.dt = CFL * self.dx / self.c_0

        self.e = np.zeros(self.x.shape)
        self.J = np.zeros(self.x.shape, dtype=np.complex_)
        self.h = np.zeros(self.xDual.shape)
     

    def get_a_p(self):
        return self.get_a_p()

    def get_c_p(self):
        return self.get_c_p()


    def step(self):
        e = self.e
        h = self.h

        cE = -self.dt / (self.dx * self.eps_0 * self.eps_inf)
        cH = -self.dt / self.dx / self.mu_0

        eMur = e[1]
        eMur2 = e[-2] 
        e[1:-1] = cE * (h[1:] - h[:-1]) + e[1:-1]

        # Lado izquierdo
        #e[0] = 0.0                                       # PEC
        # e[0] = e[0] - 2* dt/dx/eps*h[0]                  # PMC
        # e[0] =  (-dt / dx / eps) * (h[0] - h[-1]) + e[0] # Periodica
        e[0] = eMur + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[1]-e[0]) # Mur

        # Lado derecho
        #e[-1] = 0.0
        # e[-1] = e[0]
        e[-1] = eMur2 + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[-1]-e[-2]) 


        h[:] = cH * (e[1:] - e[:-1]) + h[:]

    def stepMod(self):
        e = self.e
        h = self.h
        J = self.J
        J_act = np.zeros(self.x.shape, dtype=np.complex_)
        e_act = np.zeros(self.x.shape)
        
    
        cH = -self.dt / (self.dx * self.mu_0)

        k = np.zeros(self.x.shape, dtype=np.complex_)
        beta = np.zeros(self.x.shape, dtype=np.complex_)

        k[:] = (1 + ( self.a_p[:] * self.dt ) / 2) / (1 - ( self.a_p[:] * self.dt ) / 2)
        beta[:] = (self.eps_0 * self.c_p[:] * self.dt) / (1 - ( self.a_p[:] * self.dt ) / 2)

        c_E1 = np.zeros(self.x.shape)
        c_E2 = np.zeros(self.x.shape)

        c_E1[:] = (2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) - self.sigma * self.dt) / (2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) + self.sigma * self.dt)
        c_E2[:] = (2 * self.dt) / ((2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) + self.sigma * self.dt) )

        eMur = e[1]
        eMur2 = e[-2]    
        e_act[1:-1] = - c_E2[1:-1] * (h[1:] - h[:-1]) / self.dx + c_E1[1:-1] * e[1:-1] - c_E2[1:-1] * np.real((1 + k[1:-1]) * J[1:-1])
        J_act[1:-1] = k[1:-1] * J[1:-1] + beta[1:-1] * (e_act[1:-1] - e[1:-1]) / self.dt

        e[1:-1] = e_act[1:-1]
        J[1:-1] = J_act[1:-1]

        # Lado izquierdo
        e[0] = 0.0                                       # PEC
        # e[0] = e[0] - 2* dt/dx/eps*h[0]                  # PMC
        # e[0] =  (-dt / dx / eps) * (h[0] - h[-1]) + e[0] # Periodica
        e[0] = eMur + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[1]-e[0]) # Mur

        # Lado derecho
        #e[-1] = eMur2 - (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[-1]-e[-2]) 
        e[-1] = eMur2 + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[-1]-e[-2]) 

        h[:] = cH * (e[1:] - e[:-1]) + h[:]

    def theoretical_test(self, w):
        eps_c = self.eps_0 * self.eps_inf + self.eps_0 * self.c_p_t / (w * 1j - self.a_p_t) + self.eps_0 * np.conj(self.c_p_t) / (w * 1j -  np.conj(self.a_p_t)) 
        #eps_c = np.abs(eps_c)

        return eps_c