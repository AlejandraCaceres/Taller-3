#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d
import urllib 

DD = urllib.request.urlopen("https://raw.githubusercontent.com/AlejandraCaceres/Signal2/master/signal.dat")
DD2 = urllib.request.urlopen("https://raw.githubusercontent.com/AlejandraCaceres/Incompletos2/master/incompletos.dat")

datos_senal = np.genfromtxt(DD, delimiter=',', usecols=(0,1))
datos_incompletos = np.genfromtxt(DD2, delimiter=',', usecols=(0,1))

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

print("Frecuencias calculadas con codigo propio teniendo en cuenta la resolucion de frecuencia y al frecuencia de Nyquist")



fig, ax = plt.subplots(figsize=(10,6))
ax.plot(datos_senal[:,0], datos_senal[:,1])
ax.set_xlabel('t')
plt.grid()
fig.savefig('CaceresAlejandra_signal.pdf', type='pdf')



fig, ax = plt.subplots(figsize=(10,6))
ax.plot(frecuencias, np.abs(transformada))
ax.set_yscale('log')
ax.set_xlim([-3000,3000])
ax.set_xlabel('Frecuencia')
fig.savefig('CaceresAlejandra_TF.pdf', type='pdf')
plt.grid()
print("Frecuencias Principales: Todos los armonicos por encima de 500Hz se interpretan como ruido. En el espectro aparecen 4 picos importantes en 20Hz, 160Hz, 230Hz, 263Hz y 403Hz")

