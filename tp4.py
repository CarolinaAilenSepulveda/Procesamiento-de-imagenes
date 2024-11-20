#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti
from butterworth import Butter
import warnings
import cv2
from scipy import signal
warnings.filterwarnings('ignore')

#Para realizar los graficos en la misma linea de codigo.
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

#Función para calcular el error medio cuadrático.
def rmse(img1, img2):
    return np.sqrt(np.mean((img1-img2)**2))

#Función para calcular el error medio cuadrático (%)
def rmse_per(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_val = np.max([np.max(img1), np.max(img2)])
    return np.sqrt(mse) / max_val * 100

#Cargar imagen
img_rgb = imageio.imread('imageio:chelsea.png')/255

#Para la FFT usamos una imagen en grises (tomamos el canal Y)
img = rgb2yiq(img_rgb)[:,:,0]
img_yiq = rgb2yiq(img_rgb)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[1].imshow(img, 'gray')

# Hago la Transformada de Fourier en 2D
x = np.fft.fft2(img)
# Corro el (0,0) al centro de la imagen
x = np.fft.fftshift(x)

x_module = np.abs(x)   # Modulo
x_phase = np.angle(x)  # Fase

# Aplico una escala logarítmica para poder ver mejor el modulo
x_module = np.log(x_module)

# Normalizo con el valor maximo el módulo
minimo = x_module.min()
maximo = x_module.max()
x_module = (x_module - minimo) / (maximo - minimo)

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].imshow(img,'gray')
axes[0].set_title('Imagen original')
axes[1].imshow(x_module,'gray')
axes[1].set_title('Module')
axes[2].imshow(x_phase,'gray')
axes[2].set_title('Phase')

# Guardar la imagen sin compresión
plti.imsave("x_module_mod.png",x_module,cmap='gray')
x_module = imageio.imread("x_module_mod.png")[:,:,0]/255

#Filtro pasabajos
fs = 1000  #Frecuencia de muestreo
fc = 30  #Frecuencia de corte
w = fc / (fs / 2) #Normalizo la frecuencia de corte
b, a = signal.butter(5, w, 'low')
y_filtrada = signal.filtfilt(b, a, x_module)

plt.imshow(y_filtrada,'gray')

# Leo el espectro modificado
x_module = imageio.imread("x_module_mod.png")[:,:,0]/255
# Desnormalizo con el valor maximo
x_module_back = x_module * (maximo-minimo) + minimo
# Quito la escala logaritmica que agregué anteriormente.
x_module_back = np.exp(x_module_back)
# Armo la TF nuevamente: module . e^(j.phase) (O sea junto el modulo y fase; armo el n° complejo de nuevo)
x_back = x_module_back * np.exp(1j * x_phase)
# Hago el shift inverso del (0,0)
x_back = np.fft.ifftshift(x_back)

# Hago la Transformada Inversa de Fourier
img_back = np.fft.ifft2(x_back)

fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].imshow(img,'gray')
axes[0].set_title('Imagen original')
axes[1].imshow(np.abs(img_back),'gray')
axes[1].set_title('Imagen a recuperada con IFFT')

#Función para calcular el error medio cuadrático
error_medio = rmse(img, img_back)
error_medio_porcentaje = rmse_per(img, img_back)
print("Error medio:" + str(error_medio))
print("Error medio en porcentaje:" + str(error_medio_porcentaje))