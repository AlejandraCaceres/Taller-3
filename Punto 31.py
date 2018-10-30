#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import ifft2, fft2, fftshift, fftfreq
from scipy import misc




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

Otra2Dir = if((1.2<(X**2+Y**2)**0.5 &&  (X**2+Y**2)**0.5 < 1.25))
#Eliminar Puntos
FourierA[Otra2Dir] = FourierA[Otra2Dir]/100






# In[ ]:




