import numpy as np
import matplotlib.pyplot as plt

class FDTD_Maxwell_1D():
    def __init__(self, L=10, CFL=1.0, Nx=101):
        #Dual space and normal space definition
        self.x = np.linspace(0, L, num=Nx)
        self.xDual = (self.x[1:] + self.x[:-1])/2
       
        #widht of the dispersive medium
        self.d = 0.4
        
        #Material properties:
        self.eps_0 = 1.0
        self.eps_inf = 1.0
        self.mu_0 = 1.0
        self.eta_0 = np.sqrt(self.mu_0 / self.eps_0)
        self.c_0 = 1/np.sqrt(self.eps_0*self.mu_0)
        self.sigma = 0

        #Poles of eps(w) for each point in space        
        self.a_p = np.zeros(self.x.shape, dtype=np.complex_)
        self.c_p = np.zeros(self.x.shape, dtype=np.complex_)

        #Poles of eps(w)       
        self.a_p_t = -1 + -1j
        self.c_p_t = 1 + 0j

        #Space and Time step
        self.dx = self.x[1] - self.x[0]
        self.dt = CFL * self.dx / self.c_0

        #Initial current, electric field and magnetic field
        self.e = np.zeros(self.x.shape)
        self.J = np.zeros(self.x.shape, dtype=np.complex_)
        self.h = np.zeros(self.xDual.shape)
     

    #Function to update E and H in accordance to vacuum
    def step(self):
        #Get the E and H from object
        e = self.e
        h = self.h

        #Calculate evolution constants
        cE = -self.dt / (self.dx * self.eps_0 * self.eps_inf)
        cH = -self.dt / self.dx / self.mu_0

        #Get necessary E to apply mur conditions
        eMur = e[1]
        eMur2 = e[-2] 

        #Update E
        e[1:-1] = cE * (h[1:] - h[:-1]) + e[1:-1]

        #Impose Mur conditions in both borders
        e[0] = eMur + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[1]-e[0]) 
        e[-1] = eMur2 + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[-1]-e[-2]) 

        #Update H
        h[:] = cH * (e[1:] - e[:-1]) + h[:]


    #Function to update E, H and J in accordance to a dispersive material
    def stepMod(self):
        #Get the E and H from object
        e = self.e
        h = self.h
        J = self.J

        #Define arrays to store update E and J
        J_act = np.zeros(self.x.shape, dtype=np.complex_)
        e_act = np.zeros(self.x.shape)
        
        #Calculate all necesary constants for applying the dispersive method
        cH = -self.dt / (self.dx * self.mu_0)
        k = np.zeros(self.x.shape, dtype=np.complex_)
        beta = np.zeros(self.x.shape, dtype=np.complex_)
        k[:] = (1 + ( self.a_p[:] * self.dt ) / 2) / (1 - ( self.a_p[:] * self.dt ) / 2)
        beta[:] = (self.eps_0 * self.c_p[:] * self.dt) / (1 - ( self.a_p[:] * self.dt ) / 2)
        c_E1 = np.zeros(self.x.shape)
        c_E2 = np.zeros(self.x.shape)
        c_E1[:] = (2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) - self.sigma * self.dt) / (2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) + self.sigma * self.dt)
        c_E2[:] = (2 * self.dt) / ((2 * self.eps_0 * self.eps_inf + 2 * np.real(beta[:]) + self.sigma * self.dt) )

        #Get necessary E to apply mur conditions
        eMur = e[1]
        eMur2 = e[-2]    

        #Update E and J
        e_act[1:-1] = - c_E2[1:-1] * (h[1:] - h[:-1]) / self.dx + c_E1[1:-1] * e[1:-1] - c_E2[1:-1] * np.real((1 + k[1:-1]) * J[1:-1])
        J_act[1:-1] = k[1:-1] * J[1:-1] + beta[1:-1] * (e_act[1:-1] - e[1:-1]) / self.dt
        e[1:-1] = e_act[1:-1]
        J[1:-1] = J_act[1:-1]

        #Impose Mur conditions in both borders
        e[0] = eMur + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[1]-e[0]) # Mur
        e[-1] = eMur2 + (self.c_0*self.dt-self.dx)/(self.c_0*self.dt+self.dx)*(e[-1]-e[-2]) 

        #Update H
        h[:] = cH * (e[1:] - e[:-1]) + h[:]


    #Function to calculate theoretical value for Eps
    def theoretical_test(self, w):
        eps_c = self.eps_0 * self.eps_inf + self.eps_0 * self.c_p_t / (w * 1j - self.a_p_t) + self.eps_0 * np.conj(self.c_p_t) / (w * 1j -  np.conj(self.a_p_t)) 
        return eps_c