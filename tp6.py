#Librerias
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti
import cv2

import warnings
warnings.filterwarnings('ignore')

from skimage.feature import canny #Para el filtro de bordes (mejor que el sobel del TP 5)

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

#-----------------------------------------
#Implements a general morphologic operation.
def morph_op(img, se, fcn): #fcn es la funcion que quiero hacer. Por ej -> np.max: Dilatación, np.min: Erosión
    se_flip = np.flip(se, axis=[0,1])
    rk, ck = se_flip.shape
    img_pad = np.pad(img, ((rk//2, rk//2), (ck//2, ck//2)), 'edge')
    img_out = np.zeros(img.shape)
    for r,c in np.ndindex(img.shape):
        img_out[r,c] = fcn(img_pad[r:r+rk,c:c+ck][se_flip])
    return img_out

#-----------------------------------------

def im_dilatacion(img, se):
    return morph_op(img,se,np.max)

def im_erosion(img, se):
    return morph_op(img,se,np.min)

def im_mediana(img, se):
        return morph_op(img, se, np.median)

#Cargo una imagen
img_rgb = imageio.imread('imageio:chelsea.png')/255
img_yiq = rgb2yiq(img_rgb)
img_gray = img_yiq[:,:,0]
img_bin  = canny(img_gray, sigma=2)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_rgb)
axes[1].imshow(img_gray, 'gray')
axes[2].imshow(img_bin, 'gray')

#Elemento estructurante: BOX
N=3
se_box = np.ones((N,N), dtype=bool)
print(se_box)
print()
#se_box7 = np.ones((7,7), dtype=bool)

#Elemento estructurante: CIRCLE
se_circle = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]])
se_circle = se_circle > 0
print(se_circle)

img_gray_proc = im_dilatacion(img_gray,se_box)
img_bin_proc  = im_dilatacion(img_bin, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_proc, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_proc, 'gray')

img_gray_proc = im_erosion(img_gray,se_box)
img_bin_proc  = im_erosion(img_bin, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_proc, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_proc, 'gray')

img_gray_proc = im_mediana(img_gray,se_box)
img_bin_proc  = im_mediana(img_bin, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_proc, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_proc, 'gray')

#Dilatación menos imagen
img_gray_proc = im_dilatacion(img_gray,se_box)
img_bin_proc  = im_dilatacion(img_bin, se_box)

img_gray_final = img_gray_proc - img_gray
img_bin_final = img_bin_proc - img_bin

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_final, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_final, 'gray')

#Imagen menos erosión
img_gray_proc = im_erosion(img_gray,se_box)
img_bin_proc  = im_erosion(img_bin, se_box)

img_gray_final = img_gray - img_gray_proc
img_bin_final = img_bin - img_bin_proc

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_final, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_final, 'gray')

#Dilatación menos erosión
img_gray_proc_e = im_erosion(img_gray,se_box)
img_bin_proc_e  = im_erosion(img_bin, se_box)

img_gray_proc_d = im_dilatacion(img_gray,se_box)
img_bin_proc_d  = im_dilatacion(img_bin, se_box)

img_gray_final = img_gray_proc_d - img_gray_proc_e
img_bin_final = img_bin_proc_d - img_bin_proc_e

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_final, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_final, 'gray')

#Dilatación de erosión de X
img_gray_proc_e = im_erosion(img_gray,se_box)
img_bin_proc_e  = im_erosion(img_bin, se_box)

img_gray_proc_apertura = im_dilatacion(img_gray_proc_e,se_box)
img_bin_proc_apertura  = im_dilatacion(img_bin_proc_e, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_proc_apertura, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_proc_apertura, 'gray')

#Erosión de la dilatación de X
img_gray_proc_d = im_dilatacion(img_gray,se_box)
img_bin_proc_d  = im_dilatacion(img_bin, se_box)

img_gray_proc_cierre = im_erosion(img_gray_proc_d,se_box)
img_bin_proc_cierre  = im_erosion(img_bin_proc_d, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_proc_cierre, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_proc_cierre, 'gray')

#Imagen menos apertura
tophat_gray = img_gray - img_gray_proc_apertura
tophat_bin = img_bin - img_bin_proc_apertura

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(tophat_gray, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(tophat_bin, 'gray')

#Cierre menos imagen
bothat_gray = img_gray_proc_cierre - img_gray
bothat_bin = img_bin_proc_cierre - img_bin

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(bothat_gray, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(bothat_bin, 'gray')

#Cierre de apertura de x
#La apertura ya la tengo guardada en dos variables
#Para calcular el cierre necesito la erosión de la dilatación de x, es decir debo calcular la dilatación de la apertura y luego
#aplicarle la erosión
#Erosión de la dilatación de X
img_gray_ape_d = im_dilatacion(img_gray_proc_apertura,se_box)
img_bin_ape_d  = im_dilatacion(img_bin_proc_apertura, se_box)

img_gray_ape_e= im_erosion(img_gray_ape_d,se_box)
img_bin_ape_e  = im_erosion(img_bin_ape_d, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_ape_e, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_ape_e, 'gray')

#Apertura del cierre de x
#Para calcular la apertura necesito la dilatación de la erosión de x, es decir debo calcular la erosión del cierre y luego
#aplicarle la dilatación
img_gray_ci_e = im_erosion(img_gray_proc_cierre,se_box)
img_bin_ci_e  = im_erosion(img_bin_proc_cierre, se_box)

img_gray_ci_d= im_dilatacion(img_gray_ci_e,se_box)
img_bin_ci_d  = im_dilatacion(img_bin_ci_e, se_box)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray, 'gray')
axes[0,1].imshow(img_gray_ci_d, 'gray')
axes[1,0].imshow(img_bin, 'gray')
axes[1,1].imshow(img_bin_ci_d, 'gray')

path = "/content/iris.jpg"
img_iris = cv2.imread(path)
plt.imshow(img_iris)

img_rgb_i = img_iris/255
img_yiq_i = rgb2yiq(img_rgb_i)
img_gray_i = img_yiq_i[:,:,0]
img_bin_i  = canny(img_gray_i, sigma=2)

#Aplico Top Hat
#Box
N=7
se_box = np.ones((N,N), dtype=bool)

#Dilatación de erosión de X = Apertura
img_gray_iris_e = im_erosion(img_gray_i,se_box)
img_bin_iris_e  = im_erosion(img_bin_i, se_box)

img_gray_iris_apertura = im_dilatacion(img_gray_iris_e,se_box)
img_bin_iris_apertura  = im_dilatacion(img_bin_iris_e, se_box)

tophat_gray_i = img_gray_i - img_gray_iris_apertura
tophat_bin_i = img_bin_i - img_bin_iris_apertura

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('Imagen Original')
axes[0,1].set_title('Imagen Procesada')
axes[0,0].imshow(img_gray_i, 'gray')
axes[0,1].imshow(tophat_gray_i, 'gray')
axes[1,0].imshow(img_bin_i, 'gray')
axes[1,1].imshow(tophat_bin_i, 'gray')

path = "/content/texto.jpg"
img_texto = cv2.imread(path)
plt.imshow(img_texto)

img_rgb_t = img_texto/255
img_yiq_t = rgb2yiq(img_rgb_t)
img_gray_t = img_yiq_t[:,:,0]
img_bin_t  = canny(img_gray_t, sigma=2)

#Pruebo aplicando CO
#Apertura del cierre de x

#Calculo el cierre
#Erosión de la dilatación de X
img_gray_texto_d = im_dilatacion(img_gray_t,se_circle)

img_gray_texto_cierre = im_erosion(img_gray_texto_d,se_circle)

#Para calcular la apertura necesito la dilatación de la erosión de x, es decir debo calcular la erosión del cierre y luego
#aplicarle la dilatación
img_gray_texto_e = im_erosion(img_gray_texto_cierre,se_circle)
img_gray_cit_d= im_dilatacion(img_gray_texto_e,se_circle)

fig, axes = plt.subplots(1,2, figsize=(10,10))
axes[0].set_title('Imagen Original')
axes[1].set_title('Imagen Procesada')
axes[0].imshow(img_gray_t, 'gray')
axes[1].imshow(img_gray_cit_d, 'gray')

