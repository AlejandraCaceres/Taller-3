#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Cargar todos los datos como strings y como numeros
Datos_str = np.genfromtxt('WDBC.dat', dtype=str)
Datos_num = np.genfromtxt('WDBC.dat', dtype=float, delimiter=',')

# En la lista Benigno guardo True o False dependiendo del diagnostico
n = len(Datos_str) # numero de datos
Benigno = []
for i in range(n):
    if Datos_str[i][7]=='B': # El 7 es porque B o M es el septimo caracter
        Benigno.append(True)
    else:
        Benigno.append(False)

