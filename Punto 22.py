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
        
def covarianza(X, n):
    # X son los datos
    # n es el numero de variables
    
    COV = np.zeros((n,n)) # Matriz de ceros 
    
    for i in range(n):
        for j in range(i,n): 
            Var = np.mean( (X[:,i]-np.mean(X[:,i])) * (X[:,j]-np.mean(X[:,j])) )
            # guardar el valor de var en [i,j] y [j,i] porque la matriz es simetrica
            COV[i,j] = Var
            COV[j,i] = Var 
            
    return COV

