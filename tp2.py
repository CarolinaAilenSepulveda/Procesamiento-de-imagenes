#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#Funciones propuestas en el TP
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


#Cargo dos imagenes para las operaciones
img_1 = imageio.imread('imageio:chelsea.png')/255
img_2 = imageio.imread('imageio:wikkie.png')/255

#Recorto el tamaño de la Imagen 2 para que quede igual a Imagen 1
img_2 = img_2[0:300,0:451,:]

#Muestro las imagenes
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(img_1)
axes[0].set_title('Imagen 1')
axes[1].imshow(img_2)
axes[1].set_title('Imagen 2')

#Tamaño de las imágenes
print("IMAGEN 1: ",img_1.shape)
print("IMAGEN 2: ",img_2.shape)

#Suma clampeada en RGB
img_suma = (img_1 + img_2)

#Clampeo entre 0 y 1
img_suma = np.clip(img_suma, 0, 1)
plt.imshow(img_suma)

#Resta clampeada en RGB
img_resta = (img_1 - img_2)

#Clampeo entre 0 y 1
img_resta = np.clip(img_resta, 0, 1)
plt.imshow(img_resta)

#Suma promediada en RGB
img_suma = (img_1 + img_2)/2
plt.imshow(img_suma)

#Resta promediada en RGB
img_resta = (img_1 - img_2)/2
plt.imshow(img_resta)

#Multiplico la imagen por la matriz de transformación y la separo en tres matrices independientes y,q,i
img_1_yiq = rgb2yiq(img_1)
y1 = img_1_yiq[:,:,0]
i1 = img_1_yiq[:,:,1]
q1 = img_1_yiq[:,:,2]

img_2_yiq = rgb2yiq(img_2)
y2 = img_2_yiq[:,:,0]
i2 = img_2_yiq[:,:,1]
q2 = img_2_yiq[:,:,2]

#Muestro la imagen
fig, axes = plt.subplots(2,3,figsize=(10,5))
axes[0,0].imshow(y1,'gray')
axes[0,0].set_title('Canal Y Imagen 1')
axes[0,1].imshow(i1,'gray')
axes[0,1].set_title('Canal I Imagen 1')
axes[0,2].imshow(q1,'gray')
axes[0,2].set_title('Canal Q Imagen 1')
axes[1,0].imshow(y2,'gray')
axes[1,0].set_title('Canal Y Imagen 2')
axes[1,1].imshow(i2,'gray')
axes[1,1].set_title('Canal I Imagen 2')
axes[1,2].imshow(q2,'gray')
axes[1,2].set_title('Canal Q Imagen 2')

#Suma de los canales Y (clampeada)
img_suma_y = y1 + y2
img_suma_y = np.clip(img_suma_y, 0, 1)
#Suma de los canales IQ (interpolación)
img_suma_i = (y1*i1+y2*i2)/(y1+y2)
img_suma_q = (y1*q1 + y2*q2)/(y1+y2)

#Muestro la imagen
fig, axes = plt.subplots(1,3,figsize=(10,5))
axes[0].imshow(img_suma_y,'gray')
axes[0].set_title('Canal Y Imagen Suma')
axes[1].imshow(img_suma_i,'gray')
axes[1].set_title('Canal I Imagen Suma')
axes[2].imshow(img_suma_q,'gray')
axes[2].set_title('Canal Q Imagen Suma')

#Armo la imagen con los tres canales para volver a RGB
img_suma_yiq = np.zeros(img_1_yiq.shape)
img_suma_yiq[:,:,0] = img_suma_y
img_suma_yiq[:,:,1] = img_suma_i
img_suma_yiq[:,:,2] = img_suma_q

#Vuelvo a convertir la imagen YIQ a RGB
img_suma_resultado = yiq2rgb(img_suma_yiq)
plt.imshow(img_suma_resultado)

#If lighter
#Los limites de los pixels van de 299 en x, y 450 en y
print("IMAGEN 1: ",y1.shape)
print("IMAGEN 2: ",y2.shape)
lista_x = list(range(0, 300))
lista_y = list(range(0, 451))

lig= np.zeros(img_1.shape)

for x in lista_x:
  for y in lista_y:
    if y1[x,y] > y2[x,y]:
      lig[x,y,0] = img_1[x,y,0]
      lig[x,y,1] = img_1[x,y,1]
      lig[x,y,2] = img_1[x,y,2]
    else:
      lig[x,y,0] = img_2[x,y,0]
      lig[x,y,1] = img_2[x,y,1]
      lig[x,y,2] = img_2[x,y,2]

plt.imshow(lig)

#If darker
#Los limites de los pixels van de 299 en y, y 450 en x
print("IMAGEN 1: ",y1.shape)
print("IMAGEN 2: ",y2.shape)
lista_x = list(range(0, 300))
lista_y = list(range(0, 451))

dark= np.zeros(img_1.shape)

for x in lista_x:
  for y in lista_y:
    if y1[x,y] < y2[x,y]:
      dark[x,y,0] = img_1[x,y,0]
      dark[x,y,1] = img_1[x,y,1]
      dark[x,y,2] = img_1[x,y,2]
    else:
      dark[x,y,0] = img_2[x,y,0]
      dark[x,y,1] = img_2[x,y,1]
      dark[x,y,2] = img_2[x,y,2]

plt.imshow(dark)

