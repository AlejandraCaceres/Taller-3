import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d
import urllib2

DD = urllib2.urlopen("https://raw.githubusercontent.com/AlejandraCaceres/Signal2/master/signal.dat")
DD2 = urllib2.urlopen("https://raw.githubusercontent.com/AlejandraCaceres/Incompletos2/master/incompletos.dat")

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



n = 512
xmin = min(datos_incompletos[:,0])
xmax = max(datos_incompletos[:,0])

# X uniforme
x_completos = np.linspace(xmin, xmax, n)

# y uniforme
y_completos = np.zeros((n,2))
y_completos[:,0] = Fq(x_completos)
y_completos[:,1] = Fc(x_completos)

# datos basicos de la senal
periodo = x_completos[1]-x_completos[0]
frecuencia = 1/periodo
frecuencias_completo = np.linspace(-frecuencia/2, frecuencia/2, n) # vector de frecuencias



# crear arreglos con todos los valores de k y n
kk,nn = np.meshgrid(np.arange(n), np.arange(n))
theta = -2*np.pi*kk*nn/n # Theta es lo que va en la exponencial

transformada_completos = np.zeros((n,2), dtype=complex)
transformada_completos[:,0] = np.sum(y_completos[:,0]*np.exp(1j*theta) ,1) # hacer las sumas de la transformada
transformada_completos[:,1] = np.sum(y_completos[:,1]*np.exp(1j*theta) ,1) # hacer las sumas de la transformada

transformada_completos = fftshift(transformada_completos, axes=0) # rotar el arreglo para que coincida con las frecuencias


fig, ax = plt.subplots(figsize=(10,6), nrows=3, ncols=1, sharex=True, sharey=True)
ax[0].plot(frecuencias, np.abs(transformada))
ax[0].set_yscale('log')
ax[0].set_xlim([-3000,3000])
ax[0].grid()

ax[1].plot(frecuencias_completo, np.abs(transformada_completos[:,0]))
ax[1].set_yscale('log')
ax[1].grid()

ax[2].plot(frecuencias_completo, np.abs(transformada_completos[:,1]))
ax[2].set_yscale('log')
ax[2].set_xlabel('Frecuencia')
ax[2].grid()
fig.savefig('CaceresAlejandra_TF_interpola', type='pdf')

# filtrar la senal
transformada1000 = transformada
transformada500 = transformada
transformada_completos1000 = transformada_completos
transformada_completos500 = transformada_completos

# Filtrar y hacer transformada inversa
transformada1000[np.abs(frecuencias)>1000] = 0
datos_senal_filtrada1000 = ifft(fftshift(transformada1000))

transformada500[np.abs(frecuencias)>500] = 0
datos_senal_filtrada500 = ifft(fftshift(transformada500))

transformada_completos1000[np.abs(frecuencias_completo)>1000, :] = 0
datos_completos_filtrada1000 = ifft(fftshift(transformada_completos1000), axis=0)

transformada_completos500[np.abs(frecuencias_completo)>500, :] = 0
datos_completos_filtrada500 = ifft(fftshift(transformada_completos500), axis=0)

# grafica
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(datos_senal[:,0], np.real(datos_senal_filtrada1000))
ax.set_xlabel('tiempo')
fig.savefig('CaceresAlejandra_filtrada.pdf', type='pdf')

print("La transformada de Fourier de los datos incompletos no se puede hacer porque en la formula el termino de la exponencial hace referencia a los multiplos de las frecuencias que se estan midiendo. Si la frecuencia de muestreo no es constante, esta formula no se puede aplicar")



