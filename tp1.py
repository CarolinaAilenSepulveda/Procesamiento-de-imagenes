#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline
def apply_matrix(img, M):
        return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

#Cargar imagen y analizarla como matriz
img_rgb = imageio.imread('imageio:chelsea.png')/255
plt.imshow(img_rgb)

print('Tamaño de la imagen RGB' )
print(img_rgb.shape)
print('Imagen como matriz')
print(img_rgb)
print('Primera capa de la matriz')
print(img_rgb[:,:,1])

path = "/content/lavanda.jpg"
img_iris = cv2.imread(path)
img_rgb_i = img_iris/255
plt.imshow(img_rgb_i)

#Canales RGB
#Matriz de transformación para obtener cada canal
#Matriz para obtener el canal color verde
M = np.array([[0,0,0],          # [R,0,0]
              [0,1,0],          # [0,G,0]
              [0,0,0]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb_i, M)
plt.imshow(img_red)

#Matriz para obtener el canal color azul
M = np.array([[0,0,0],          # [R,0,0]
              [0,0,0],          # [0,G,0]
              [0,0,1]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb_i, M)
plt.imshow(img_red)

#Canales RGB
#Matriz de transformación para obtener cada canal
#Matriz para obtener el canal color verde
M = np.array([[0,0,0],          # [R,0,0]
              [0,1,0],          # [0,G,0]
              [0,0,0]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb, M)
plt.imshow(img_red)

#Matriz para obtener el canal color rojo
M = np.array([[1,0,0],          # [R,0,0]
              [0,0,0],          # [0,G,0]
              [0,0,0]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb, M)
plt.imshow(img_red)

#Matriz para obtener el canal color azul
M = np.array([[0,0,0],          # [R,0,0]
              [0,0,0],          # [0,G,0]
              [0,0,1]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb, M)
plt.imshow(img_red)

M = np.array([[0,1,0],          # [R,0,0]
              [0,0,1],          # [0,G,0]
              [1,0,0]])         # [0,0,B]

#Multiplico la imagen por la matriz de transformación
img_red = apply_matrix(img_rgb, M)
plt.imshow(img_red)

img_rgb[:,:,0] = img_rgb[:,:,1]
img_rgb[:,:,1] = img_rgb[:,:,2]
img_rgb[:,:,2] = img_rgb[:,:,0]
plt.imshow(img_rgb)

#Matriz de transformación.
M_YIQ = np.array([[0.299, 0.587, 0.114],
                  [0.595716, -0.274453, -0.321263],
                  [0.211456, -0.522591, 0.311135]])

#Multiplico la imagen por la matriz de transformación y la separo en tres matrices independientes.
img_yiq = apply_matrix(img_rgb, M_YIQ)
y_channel = img_yiq[:,:,0]
i_channel = img_yiq[:,:,1]
q_channel = img_yiq[:,:,2]

#Muestro la imagen
fig, axes = plt.subplots(1,3,figsize=(10,5))
axes[0].imshow(y_channel,'gray')
axes[0].set_title('Canal Y')
axes[1].imshow(i_channel,'gray')
axes[1].set_title('Canal I')
axes[2].imshow(q_channel,'gray')
axes[2].set_title('Canal Q')

#Multiplico los canales por una constante
y_channel_mod = y_channel*3
i_channel_mod = i_channel*1
q_channel_mod = q_channel*1

#Recorto entre min() y 1 para plotearlas. Pueden ser negativos, por eso no recorto en 0.
y_channel_mod = np.clip(y_channel_mod,y_channel_mod.min(),1)
i_channel_mod = np.clip(i_channel_mod,i_channel_mod.min(),1)
q_channel_mod = np.clip(q_channel_mod,q_channel_mod.min(),1)

#Muestro la imagen
fig, axes = plt.subplots(1,3,figsize=(10,5))
axes[0].imshow(y_channel_mod,'gray')
axes[0].set_title('Canal Y')
axes[1].imshow(i_channel_mod,'gray')
axes[1].set_title('Canal I')
axes[2].imshow(q_channel_mod,'gray')
axes[2].set_title('Canal Q')

#Multiplico los canales por una constante
y_channel_mod = y_channel*2
i_channel_mod = i_channel*6
q_channel_mod = q_channel*10

#Recorto entre min() y 1 para plotearlas. Pueden ser negativos, por eso no recorto en 0.
y_channel_mod = np.clip(y_channel_mod,y_channel_mod.min(),1)
i_channel_mod = np.clip(i_channel_mod,i_channel_mod.min(),1)
q_channel_mod = np.clip(q_channel_mod,q_channel_mod.min(),1)

#Muestro la imagen
fig, axes = plt.subplots(1,3,figsize=(10,5))
axes[0].imshow(y_channel_mod,'gray')
axes[0].set_title('Canal Y')
axes[1].imshow(i_channel_mod,'gray')
axes[1].set_title('Canal I')
axes[2].imshow(q_channel_mod,'gray')
axes[2].set_title('Canal Q')

M_RGB = np.array([[1, 0.9563, 0.6210],
                  [1, -0.2721, -0.6474],
                  [1, -1.1070, 1.7046]])

M_RGB_modificada = np.array([[0.1, 0.47058824, 0.46058824],
                  [0.48235294,-0.1,0.45684],
                  [0.485, -0.19411765,0.55132]])

#Creo una imagen vacia del mismo tamaño que la original (y con 3 canales)
img_yiq_mod = np.zeros(img_yiq.shape) #print(img_yiq_mod.shape)

#Guardo cada canal en la imagen vacia
img_yiq_mod[:,:,0] = y_channel_mod
img_yiq_mod[:,:,1] = i_channel_mod
img_yiq_mod[:,:,2] = q_channel_mod

#Multiplico la imagen por la matriz de transformación.
img_rgb_mod = apply_matrix(img_yiq_mod, M_RGB)
img_rgb_mod_2 = apply_matrix(img_yiq_mod, M_RGB_modificada)

fig, axes = plt.subplots(1,3,figsize=(10,5))
axes[0].imshow(np.clip(img_rgb_mod,0,1))
axes[0].set_title('Imagen con YIQ modificado')
axes[1].imshow(np.clip(img_rgb,0,1))
axes[1].set_title('Imagen Original')
axes[2].imshow(np.clip(img_rgb_mod_2,0,1))
axes[2].set_title('Imagen con matriz modificada')

M_RGB = np.array([[1, 0.9563, 0.6210],
                  [1, -0.2721, -0.6474],
                  [1, -1.1070, 1.7046]])

#Creo una imagen vacia del mismo tamaño que la original (y con 3 canales)
img_yiq_mod = np.zeros(img_yiq.shape) #print(img_yiq_mod.shape)

#Guardo cada canal en la imagen vacia
img_yiq_mod[:,:,0] = y_channel_mod
img_yiq_mod[:,:,1] = i_channel_mod
img_yiq_mod[:,:,2] = q_channel_mod

#Multiplico la imagen por la matriz de transformación.
img_rgb_mod = apply_matrix(img_yiq, M_RGB)

fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(np.clip(img_rgb_mod,0,1))
axes[0].set_title('Imagen con YIQ original')
axes[1].imshow(np.clip(img_rgb,0,1))
axes[1].set_title('Imagen Original')

