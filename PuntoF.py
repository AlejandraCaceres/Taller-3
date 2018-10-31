#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d

datos_senal = np.genfromtxt('signal.dat', delimiter=',', usecols=(0,1))
datos_incompletos = np.genfromtxt('incompletos.dat', delimiter=',', usecols=(0,1))

# FUnciones de interpolacion
Fq = interp1d(datos_incompletos[:,0], datos_incompletos[:,1], kind='quadratic')
Fc = interp1d(datos_incompletos[:,0], datos_incompletos[:,1], kind='cubic')

# datos basicos de la senal
periodo = datos_senal[1,0]-datos_senal[0,0]
frecuencia = 1/periodo
n = len(datos_senal[:,0]) # numero de datos
frecuencias = np.linspace(-frecuencia/2, frecuencia/2, n) # vector de frecuencias

# crear arreglos con todos los valores de k y n
kk,nn = np.meshgrid(np.arange(n), np.arange(n))
theta = -2*np.pi*kk*nn/n # Theta es lo que va en la exponencial
transformada = np.sum(datos_senal[:,1]*np.exp(1j*theta) ,1) # hacer las sumas de la transformada
transformada = fftshift(transformada) # rotar el arreglo para que coincida con las frecuencias


# In[ ]:





# In[ ]:




