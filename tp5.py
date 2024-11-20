#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti

import warnings
warnings.filterwarnings('ignore')

from matplotlib import cm #Para graficar los kernels
from scipy.signal import convolve2d #Función para hacer la convolución (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
import scipy.stats as st  #Para la distribucion gaussiana.

%matplotlib inline

#Funciones
#Matriz de transformación para YIQ.
M_YIQ = np.array([[0.299, 0.587, 0.114],
                  [0.595716, -0.274453, -0.321263],
                  [0.211456, -0.522591, 0.311135]])

#Matriz de transformación para RGB.
M_RGB = np.array([[1, 0.9563, 0.6210],
                  [1, -0.2721, -0.6474],
                  [1, -1.1070, 1.7046]])

#Función para operaciones con matrices.
def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

#Función simplificada para convertir RGB->YIQ.
def rgb2yiq(img):
    return apply_matrix(img, M_YIQ)

#Función simplificada para convertir YIQ->RGB.
def yiq2rgb(img):
    return apply_matrix(img, M_RGB)


#Filtro Gaussiano
def gaussian(N, sigma=1): #N: tamaño del kernel, sigma: Desviacion estandar
    x = np.linspace(-sigma, sigma, N+1) #linspace crea un vector de valores entre -sigma y sigma igualmente distribuidos
    gaussian_dist = np.diff(st.norm.cdf(x)) #CDF = Cumulative distribution function NORM: distribucion normal/gaussiana.
    gaussian_filter = np.outer(gaussian_dist, gaussian_dist)
    return gaussian_filter/gaussian_filter.sum()


#Para dibujar el Kernel en 3D.
def plot_kernel(data, ax=None):
    rows, cols = data.shape
    y, x = np.meshgrid(np.arange(rows),np.arange(cols),indexing='ij')
    if ax == None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    _min, _max = (np.min(data), np.max(data))
    ax.plot_surface(x, y, data.T, cmap=cm.jet, vmin=_min, vmax=_max)

#Para graficar imagen, imagen filtrada y kernel en un solo grafico.
def plot_images_and_kernel(img, img_filt, kernel):
    fig = plt.figure(figsize=(17,5))
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, 'gray')
    ax1.title.set_text('Input image')
    ax2 = fig.add_subplot(132)
    ax2.imshow(img_filt, 'gray')
    ax2.title.set_text('Filtered image')
    ax3 = fig.add_subplot(133, projection='3d')
    plot_kernel(kernel, ax=ax3)
    ax3.title.set_text('Kernel')
    plt.show()

    #Cargo una imagen
img_rgb = imageio.imread('imageio:chelsea.png')/255

#Usamos una imagen en grises (tomamos el canal Y)
img = rgb2yiq(img_rgb)[:,:,0]
img_yiq = rgb2yiq(img_rgb)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[1].imshow(img, 'gray')

#Filtro pasabajos cuadrado
kernel = np.array ([[1,1,1],
                    [1,1,1],
                    [1,1,1]])
# Normalizo
kernel = kernel / kernel.sum()
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro pasabajos circular
kernel = np.array ([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
# Normalizo
kernel = kernel / kernel.sum()
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro pasabajos piramidal
kernel = np.array ([[1,2,1],
                    [2,4,2],
                    [1,2,1]])
# Normalizo
kernel = kernel / kernel.sum()
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro pasabajos Gaussiano
kernel = gaussian(12,3)

print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro pasa altos Laplace 4 vecinos
kernel = np.array ([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

img_absoluta = np.abs(img_filt)

# Muestro la Convolución
plot_images_and_kernel(img, img_absoluta, kernel)
plti.imsave("image_saved.png",img_absoluta,cmap='gray')

#Filtro pasa altos Laplace 8 vecinos
kernel = np.array ([[-1,-1,-1],
                    [-1,8,-1],
                    [-1,-1,-1]])
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

img_absoluta = np.abs(img_filt)

# Muestro la Convolución
plot_images_and_kernel(img, img_absoluta, kernel)
plti.imsave("image_saved.png",img_absoluta,cmap='gray')

#Filtro pasa altos a partir de un pasabajos
#PA=ID-PB
#Trabajo con el filtro pasabajos circular
kernel_pb = np.array ([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
identidad = np.array ([[0,0,0],
                    [0,1,0],
                    [0,0,0]])
kernel = identidad - kernel_pb
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro pasa banda DoG
#El filtro pasabanda se construye como la resta de dos filtros pasabajos gaussianos
#Voy a emplear el gaussiano con desviación 3 y 5
kernel_1 = gaussian(12,3)
kernel_2 = gaussian(12,5)
kernel = kernel_1 - kernel_2
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

#Filtro mejora de contraste
#Se suma el kernel identidad más un porcentaje (por ejemplo 0,3) del pasa altos
#Empleo el filtro pasaaltos Laplace de cuatro vecinos
kernel_pasaaltos = np.array ([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])
kernel_identidad = np.array ([[0,0,0],
                    [0,1,0],
                    [0,0,0]])

kernel = kernel_identidad + (0.3*kernel_pasaaltos)
print(kernel)

# Operación de Convolución
img_filt = convolve2d(img,kernel,mode='same')

# Muestro la Convolución
plot_images_and_kernel(img, img_filt, kernel)
plti.imsave("image_saved.png",img_filt,cmap='gray')

# Filtro Sobel 3x3 (gradiente en X y gradiente en Y)
# Kernel Sobel Gradiente en X
kernel_X = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]])

# Kernel Sobel Gradiente en Y
kernel_Y = np.array([[1,2,1],
                    [0,0,0],
                    [-1,-2,-1]])

# Hago la convolución para cada filtro
Gx = convolve2d(img,kernel_X,mode='same')
Gy = convolve2d(img,kernel_Y,mode='same')

# Calculo el valor absoluto
img_filt_abs = np.sqrt(Gx*Gx + Gy*Gy)

#Grafico solo el kernel X
plot_images_and_kernel(img, img_filt_abs, kernel_X)
print("Valor absoluto de Gx y Gy: ",img_filt_abs)
print(img_filt_abs.shape)

#Aplicar un umbral a la imagen obtenida por módulo, por ejemplo 0.3
lista_x = list(range(0, 300))
lista_y = list(range(0, 451))
A = np.zeros (img_filt_abs.shape)

for x in lista_x:
  for y in lista_y:
    if img_filt_abs[x,y] < 0.3:
      A[x,y] = 0
    elif img_filt_abs[x,y] > 0.3:
      A[x,y] = 1

#Grafico solo el kernel y (el x esta graficado en el paso anterior)
plot_images_and_kernel(img,A, kernel_Y)

