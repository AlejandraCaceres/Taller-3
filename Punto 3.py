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
print (n)


# In[ ]:




