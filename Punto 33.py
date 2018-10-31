#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import ifft2, fft2, fftshift, fftfreq
from scipy import misc

# Guardo la imagen en un array
imagenA = misc.imread('arbol.PNG')
FourierA = fft2(imagenA)
FourierA = FourierA/np.max(FourierA) # Guardar imagen

fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(fftshift(np.abs(FourierA)))
fig.savefig('CaceresAlejandra_FT2D.pdf', type='pdf')

n = len(FourierA)
#print (n)

#Creo la grilla de Datos 
X,Y = np.meshgrid( np.linspace(-1,1,n), np.linspace(-1,1,n) )

# Condiciones para elegir que puntos quiero eliminar
II = np.where( np.logical_and( np.logical_and(1.2<(X**2+Y**2)**0.5, (X**2+Y**2)**0.5 < 1.25), X*Y>0 ) )
FourierA[II] = FourierA[II]/100

II = np.where( np.logical_and( np.logical_and(0.7<(X**2+Y**2)**0.5, (X**2+Y**2)**0.5 < 0.75), X*Y>0 ) )
FourierA[II] = FourierA[II]/100

# Espectro de frecuencias luego del filtro
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(fftshift(((FourierA))))
fig.savefig('CaceresAlejandra_FT2D_filtrada', type='pdf')





# In[ ]:




