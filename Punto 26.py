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

Datos_num = Datos_num[:,2:5] # Elegir solo las variables de interes
COV = covarianza(Datos_num, np.size(Datos_num,1))
print(COV, '\n\n', np.cov(Datos_num.T))

# Reordenar los arreglos para que el mayor autovector este de primero
for i in range(n):
    if valoresPropios[i] > valoresPropios[0]: # si hay uno mayor que el primero, reordenar
        # reasignar valores
        valor = valoresPropios[0]
        vector = vectoresPropios[:,0]
        
        valoresPropios[0] = valoresPropios[i]
        vectoresPropios[:,0] = vectoresPropios[:,i]
        valoresPropios[i] = valor
        vectoresPropios[:,i] = vector
        
# Reordenar los arreglos para que el mayor autovector este de primero
for i in range(1,n):
    if valoresPropios[i] > valoresPropios[1]: # si hay uno mayor que el primero, reordenar
        # reasignar valores
        valor = valoresPropios[1]
        vector = vectoresPropios[:,1]
        
        valoresPropios[1] = valoresPropios[i]
        vectoresPropios[:,1] = vectoresPropios[:,i]
        valoresPropios[i] = valor
        vectoresPropios[:,i] = vector
#Valores y vectores propios

valoresPropios, vectoresPropios = np.linalg.eig(COV)

print(" ")
print("Valores propios")
for i in range(len(COV)):
    print(i+1, "   ", valoresPropios[i])

print(" ")    
print("Vectores propios")
for i in range(len(COV)):
    print(i+1, "   ", vectoresPropios[:,i])

n = len(COV)

importante1 = np.where(abs(vectoresPropios[:,0])==max(abs(vectoresPropios[:,0])))
importante2 = np.where(abs(vectoresPropios[:,1])==max(abs(vectoresPropios[:,1])))

Datos_proyectados_c1 = np.zeros(len(Datos_num))
Datos_proyectados_c2 = np.zeros(len(Datos_num))

for i in range(len(Datos_proyectados_c1)):
    Datos_proyectados_c1[i] = np.dot(Datos_num[i,:], vectoresPropios[:,0])
    Datos_proyectados_c2[i] = np.dot(Datos_num[i,:], vectoresPropios[:,1])
    

