#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti
from scipy.signal import convolve2d
import scipy.stats as st
import cv2
%matplotlib inline

#Funciones
M_YIQ = np.array([[0.299, 0.587, 0.114],
                  [0.595716, -0.274453, -0.321263],
                  [0.211456, -0.522591, 0.311135]])

M_RGB = np.array([[1, 0.9563, 0.6210],
                  [1, -0.2721, -0.6474],
                  [1, -1.1070, 1.7046]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

def rgb2yiq(img):
    return apply_matrix(img, M_YIQ)

def yiq2rgb(img):
    return apply_matrix(img, M_RGB)

#Filtro Gaussiano
def gaussian(N, sigma=1):
    x = np.linspace(-sigma, sigma, N+1) #linspace crea un vector de valores entre -sigma y sigma igualmente distribuidos
    gaussian_dist = np.diff(st.norm.cdf(x)) #CDF = Cumulative distribution function NORM: distribucion normal/gaussiana.
    gaussian_filter = np.outer(gaussian_dist, gaussian_dist)
    return gaussian_filter/gaussian_filter.sum()

#Cargo una imagen
img_rgb  = imageio.imread('imageio:chelsea.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]               #Tomo el canal Y
print(img_gray.shape)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[1].imshow(img_gray, 'gray')

def downsamplingX2(imagen):
  return imagen[::2,::2]   #Cada pixel de salida contiene cuatro de entrada, es decir tengo que ir de a pasos de dos en altura y dos en ancho (cuatro en total)

img_mod = downsamplingX2(img_gray)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

#El primer pixel es el promedio de la cuadrícula de 2x2, me queda la mitad de la imagen
img_vacia= np.zeros((150,225))

def promedio2x2 (imagen, imagen_vacia):
  contadorx = 0
  contadory = 0
  for x in range(0, 300,2):   #Recorro la imagen original
    contadory = 0
    if x != 0:
      contadorx = contadorx + 1
    for y in range(0,450,2):
    #En la primera iteración tengo el pixel (0,0) y necesito el (0,1),(1,0),(1,1)
      if (x+1 < 300):
        if (y+1 < 451):
          #print(x,y)
          #print(contadorx, contadory)
          #print("-------------")
          imagen_vacia[contadorx,contadory] = (imagen[x,y] + imagen[x+1,y] + imagen[x,y+1] + imagen[x+1,y+1])/4
          contadory = contadory + 1
  return imagen_vacia

img_mod = promedio2x2(img_gray, img_vacia)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

#Samplig x2
img_mod = downsamplingX2(img_gray)

#Filtro pasabajos Gaussiano
kernel = gaussian(12,3)

# Operación de Convolución
img_filt = convolve2d(img_mod,kernel,mode='same')

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_filt, "gray")

#Agarro el primer pixel y lo repito en los elementos (1,1), (1,2), (2,1) y (2,2) y así sigo
#La imagen resultante es del doble size de la original

img_vacia= np.zeros((600,902))

def upsamplingx2 (imagen, imagen_vacia):
  contadorx = 0
  contadory = 0
  for x in range(0, 600,2):  #Recorro la imagen nueva (del doble size)
    if x != 0:
      contadorx = contadorx + 1
    contadory = 0
    for y in range(0,902,2):
        imagen_vacia[x:x+2,y:y+2] = imagen[contadorx,contadory]
        contadory = contadory + 1
  return imagen_vacia

img_mod = upsamplingx2(img_gray, img_vacia)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

#Resuelvo con convolución definiendo el kernel
kernel = np.array ([[1/4,1/2,1/4],
                    [1/2,1,1/2],
                    [1/4,1/2,1/4]])
# Normalizo
kernel = kernel / kernel.sum()
print(kernel)
print(img_gray.shape)

#Armo una imagen con ceros
img_vacia = np.zeros((600,902))
img_vacia[::2,::2] = img_gray

# Operación de Convolución
img_filt = convolve2d(img_vacia,kernel,mode='same')

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_filt, "gray")

#Resuelvo con convolución definiendo el kernel
vector = np.array([-1/8,0,5/8,1,5/8,0,1/8])
vector_c = np.array([[-1/8],[0],[5/8],[1],[5/8],[0],[1/8]])

#Calculo el kernel
kernel =vector * vector_c

# Normalizo
kernel = kernel / kernel.sum()
print(kernel)

#Armo una imagen con ceros
img_vacia = np.zeros((600,902))
img_vacia[::2,::2] = img_gray

# Operación de Convolución
img_filt = convolve2d(img_vacia,kernel,mode='same')

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_filt, "gray")

#Repetir pixeles en grilla 2x2
img_vacia= np.zeros((600,902))
img_mod = upsamplingx2(img_gray, img_vacia)

#Filtro pasabajos Gaussiano
kernel = gaussian(12,3)

# Operación de Convolución
img_filt = convolve2d(img_mod,kernel,mode='same')

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_filt, "gray")

#Transformada de Fourier
x = np.fft.fft2(img_gray)

# Corro el (0,0) al centro de la imagen
x = np.fft.fftshift(x)

#Si quiero aumentar en 20 la proporción (agrego ceros alrededor del centro)
x_resize = np.pad(x,(10,10), 'constant', constant_values= 0)
x_resize.shape

#Hago la transformada inversa
x_back = np.fft.ifftshift(x_resize)

# Hago la Transformada Inversa de Fourier
img_back = np.fft.ifft2(x_back)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(np.abs(img_back), "gray")

#Transformada de Fourier
x = np.fft.fft2(img_gray)

# Corro el (0,0) al centro de la imagen
x = np.fft.fftshift(x)

#Si quiero tener la mitad de shape (150, 225)
x_resize = np.resize(x,(150,225))

#Hago la transformada inversa
x_back = np.fft.ifftshift(x_resize)

# Hago la Transformada Inversa de Fourier
img_back = np.fft.ifft2(x_back)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(np.abs(img_back), "gray")

#Cuantizar un pixel en N niveles
N=5
img_mod = np.round(img_gray*(N-1))/(N-1)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

img_vacia= np.zeros((300,451))
lista_x = list(range(0, 300))
lista_y = list(range(0, 451))

def scanline(imagen,lista_y,lista_x,imagen_vacia):
  error=0
  for x in lista_x:
    for y in lista_y:
      if (type(error) == float or type(error) == int):
        imagen_vacia[x,y] = round(imagen[x,y]+ error)
        error = error + (imagen_vacia[x,y] - imagen[x,y])
      else:
        imagen_vacia[x,y] = round(imagen[x,y])
  return imagen_vacia

img_mod = scanline(img_gray, lista_y,lista_x, img_vacia)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

img_vacia= np.zeros((300,451))
lista_x = list(range(0, 299))
lista_y = list(range(0, 450))

def floyd(imagen,lista_y,lista_x,imagen_vacia,N):
  for y in lista_y:
      for x in lista_x:
          oldpixel = imagen[x,y]
          newpixel = np.round(oldpixel)/N
          imagen_vacia[x,y] = newpixel
          quant_error = oldpixel - newpixel
          imagen_vacia[x+1,y] = imagen[x + 1,y] + quant_error*(7/16)
          imagen_vacia[x-1,y+1] = imagen[x-1,y+1] + quant_error*(3/16)
          imagen_vacia[x,y+1] = imagen[x,y+1] + quant_error*(5/16)
          imagen_vacia[x + 1,y + 1] = imagen[x + 1,y + 1] + quant_error*(1/16)
  return imagen_vacia

img_mod = floyd(img_gray, lista_y,lista_x,img_vacia,3)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray, "gray")
axes[1].imshow(img_mod, "gray")

