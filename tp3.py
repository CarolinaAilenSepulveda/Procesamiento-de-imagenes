#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt

import warnings
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

#Función para graficar un histograma.
# -im: imagen de entrada.
# -bins: cantidad de barritas del histograma.
# -ax: el eje donde quiero graficar el histograma (en qué subplot graficarlo).
# -cumulative: grafico acumulado (opcional)
def plot_hist(im, bins, ax, cumulative=False):
    counts, borders = np.histogram(im if im.ndim==2 else rgb2yiq(im)[...,0], bins=bins, range=(0,1))
    ax.bar(range(len(counts)), np.cumsum(counts) if cumulative else counts) #barh para horizontal.
    plt.xticks(ax.get_xticks(), labels=np.round(ax.get_xticks()/bins,2))
    plt.grid(alpha=0.3)

#Cargar imagen y mostrar el histograma
img_rgb = imageio.imread('imageio:chelsea.png')/255
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb)
plot_hist(img_rgb, 35, axes[1])

#Para normalizar el histograma empleando una función debo pasar de RGB a YIQ, y empleando la componente de luminancia la debo normalizar
#Para esto se estira el máximo a 1 y el mínimo a cero
img_yiq = rgb2yiq(img_rgb)
y_channel = img_yiq[:,:,0]
print("Y antes de normalizar --------")
print("MAXIMO: ",y_channel.max())
print("MINIMO: ",y_channel.min())

#Para normalizar empleo la fórmula Valor_n= Valor-min/max-min
print(y_channel.shape)
A = np.ones ((300,  451))
B = np.subtract(y_channel, A*(y_channel.min()))   #Matriz Y menos el valor mínimo de la luminancia

#Ahora me falta dividir los valores por el valor máximo menos el mínimo
coeficiente = y_channel.max() - y_channel.min()
Y_normalizada = B/coeficiente

print("Y despues de normalizar -------")
print(Y_normalizada)
print(Y_normalizada.shape)
print("MAXIMO: ",Y_normalizada.max())
print("MINIMO: ",Y_normalizada.min())
#Vuelvo al espacio RGB y muestro la imagen final
#Armo la imagen con los tres canales para volver a RGB
img_norm_yiq = np.zeros(img_yiq.shape)
img_norm_yiq[:,:,0] = Y_normalizada
img_norm_yiq[:,:,1] = img_yiq[:,:,1]
img_norm_yiq[:,:,2] = img_yiq[:,:,2]

#Vuelvo a convertir la imagen YIQ a RGB
img_norm_resultado = yiq2rgb(img_norm_yiq)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_norm_resultado)
plot_hist(img_norm_resultado, 35, axes[1])

print("Y antes de normalizar --------")
print("MAXIMO: ",y_channel.max())
print("MINIMO: ",y_channel.min())
maximo = np.percentile(y_channel,99)
minimo = np.percentile(y_channel,1)
print("Percentil max:",maximo)
print("Percentil min:",minimo)

lista_x = list(range(0, 300))
lista_y = list(range(0, 451))
A = np.zeros (y_channel.shape)

for x in lista_x:
  for y in lista_y:
    if y_channel[x,y] < minimo:
      A[x,y] = 0
    elif y_channel[x,y] > maximo:
      A[x,y] = 1
    else:
      A[x,y] = (y_channel[x,y] - minimo)/maximo-minimo

print("Y despues de normalizar -------")
print(A)
print(A.shape)
print("MAXIMO: ",A.max())
print("MINIMO: ",A.min())

#Vuelvo al espacio RGB y muestro la imagen final
#Armo la imagen con los tres canales para volver a RGB
img_per_yiq = np.zeros(img_yiq.shape)
img_per_yiq[:,:,0] = A
img_per_yiq[:,:,1] = img_yiq[:,:,1]
img_per_yiq[:,:,2] = img_yiq[:,:,2]

#Vuelvo a convertir la imagen YIQ a RGB
img_per_resultado = yiq2rgb(img_per_yiq)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_per_resultado)
plot_hist(img_per_resultado, 35, axes[1])

#2.4 es el valor del factor gamma de corrección
y_gamma = y_channel**2.4
print(y_gamma)

#Vuelvo al espacio RGB y muestro la imagen final
#Armo la imagen con los tres canales para volver a RGB
img_gamma_yiq = np.zeros(img_yiq.shape)
img_gamma_yiq[:,:,0] = y_gamma
img_gamma_yiq[:,:,1] = img_yiq[:,:,1]
img_gamma_yiq[:,:,2] = img_yiq[:,:,2]

#Vuelvo a convertir la imagen YIQ a RGB
img_gamma_resultado = yiq2rgb(img_gamma_yiq)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gamma_resultado)
plot_hist(img_gamma_resultado, 35, axes[1])

#Lineal a trozos
x = np.array([0, 0.4, 0.7, 0.8,  1])
y = np.array([0, 0.05, 0.07, 0.95, 1])
y_channel_mod = np.interp(y_channel, x, y)

plt.plot(x,y)

#Vuelvo al espacio RGB
img_yiq[:,:,0] = y_channel_mod
img_rgb_mod2 = yiq2rgb(img_yiq)

#Muestro la imagen modificada
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[0].set_title('Imagen original')
axes[1].imshow(img_rgb_mod2)
axes[1].set_title('Imagen modificada')
plot_hist(img_rgb_mod2, 35, axes[2])
axes[2].set_title('Histograma imagen modificada')

#Raíz cuadrada
y_channel_mod = np.sqrt(y_channel)

#Vuelvo al espacio RGB
img_yiq[:,:,0] = y_channel_mod
img_rgb_mod2 = yiq2rgb(img_yiq)

#Muestro la imagen modificada
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[0].set_title('Imagen original')
axes[1].imshow(img_rgb_mod2)
axes[1].set_title('Imagen modificada')
plot_hist(img_rgb_mod2, 35, axes[2])
axes[2].set_title('Histograma imagen modificada')

