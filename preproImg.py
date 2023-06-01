# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:37:41 2023

@author: Carlos
"""

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import pandas as pd
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

def preProcessing(path, n, batch):
    """
    
    patch de carpeta de imagenes 
    n:
    batch: pixeles de recorte
    """
    rutaImgOr = path #'G:\Maestria\Analisis numerico\Proyecto\Imagens\img-'
    rutaImg = 'G:\Maestria\Analisis numerico\Proyecto\Test\Scaling\img-'
    typeImg = '.jpeg'
    
    for i in range(1,n+1):
        print(rutaImgOr + str(i) + typeImg)
        img = cv2.imread(rutaImgOr + str(i) + typeImg)
    
    # Reescala la imagen a 128x128
        img_resized = cv2.resize(img, (1280, 1280))
    
    # Convierte la imagen a escala de grises
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Guarda la imagen procesada
        cv2.imwrite(rutaImg + str(i) + typeImg, img_gray)
    
    
    """
    rutaImgOr = 'G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-'
    rutaImg = 'G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\img-'
    typeImg = '.jpeg'
    g = 0
    for i in range(1,n+1):
        
        img = cv2.imread(rutaImgOr + str(i) + typeImg)
        
    # Obtiene las dimensiones de la imagen
        height, width, channels = img.shape
    
    # Calcula las coordenadas del recorte
        for j in range(0,2):
            x = int((width - batch) / 2)
            y = int((height - batch) / 2)
            w = batch
            h = batch
    
    # Recorta la imagen en el centro
            for z in range(0,2):
                img_cropped = img[y+j*h:y+h+j*h, x+z*w:x+w+z*w]
                g =g+1
    # Convierte la imagen a escala de grises
                img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    
    # Guarda la imagen procesada
                cv2.imwrite(rutaImg + str(g)+ typeImg, img_gray)
                print(rutaImg + str(g)+ typeImg)
    """
                
def random_crops_RL(image_path, crop_size=64, num_crops=5, save_path=None):
    
    
    # cargar la imagen y convertirla en un array de NumPy
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_array = np.array(image) / 255.0
    
    # obtener las dimensiones de la imagen
    height, width = image_array.shape
    
    # calcular las coordenadas para los recortes centrados
    center_x, center_y = width // 2, height // 2
    
    crops = []
    for i in range(num_crops):
        # calcular las coordenadas aleatorias para el recorte
        x = np.random.randint(center_x - crop_size // 2, center_x + crop_size // 2)
        y = np.random.randint(center_y - crop_size // 2, center_y + crop_size // 2)
        
        # extraer el recorte de la imagen original
        crop = image_array[y - crop_size // 2:y + crop_size // 2, x - crop_size // 2:x + crop_size // 2]
        # agregar el recorte a la lista
        crops.append(crop)
        
        if save_path is not None:
            
            cv2.imwrite(save_path+str(i)+".jpeg", image[y - crop_size // 2:y + crop_size // 2, x - crop_size // 2:x + crop_size // 2])
       
    
    # convertir la lista de recortes en un array de NumPy
    crops_array = np.array(crops)
    
    return crops_array

def load_images_x_train(folder_path, n_img):
    images = []
    for i in range(1,n_img):
        img_path = folder_path+str(i)+".jpeg"
        print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # leer la imagen en escala de grises
        #cv2.imshow("imagen",image)
        #cv2.waitKey(0)
        resized_image = cv2.resize(image, (64, 64))  # reescalar la imagen a 64x64
        images.append(resized_image / 255.0)  # normalizar la imagen
    return np.array(images)

def create_model_RL(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=2, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_model_BC(input_shape):
    
    
    model = Sequential()
    
    # Capas de convolución
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape, strides=2))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', strides=2))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', strides=2))
    
    # Capa de flatten
    model.add(Flatten())
    
    # Capa densa de 6 neuronas con activación relu
    model.add(Dense(units=6, activation='relu'))
    
    # Capa densa de una neurona de salida con función sigmoide
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compilación del modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_CNN(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_model_CNN(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def load_csv(file_path, colm):
    # Cargar el archivo CSV con pandas
    data = pd.read_csv(file_path)

    # Obtener los datos de la columna especificada como el 'y_train'
    y_train = data.iloc[:, colm].values

    # Convertir los datos en un objeto de matriz NumPy
    y_train = np.array(y_train)

    return y_train

def image_to_array_crop64(image):
   
    # Obtener dimensiones de la imagen
    height, width = image.shape
    # Inicializar lista para almacenar recuadros
    crops = []
    center_coords =[]
    # Iterar sobre la imagen
    for i in range(0, height-63, 8):
        for j in range(0, width-63, 8):
            # Obtener recuadro de 64x64 pixeles
            crop = image[i:i+64, j:j+64]
            
            center_coords.append([i+32,j+32])
            # Añadir recuadro a la lista
            
            crops.append(crop)
    # Convertir lista de recuadros a un array de numpy de (n,64,64)
    
    crops_array = np.array(crops) / 255.0
    coord = np.array(center_coords)
    return crops_array, coord

def draw_points(image, points, color=(0, 0, 255), radius=10):
    """
    Dibuja un punto en cada una de las coordenadas especificadas en la imagen cv2.
    Args:
        image: imagen en formato cv2
        points: vector de coordenadas de la forma [(x1,y1),(x2,y2),...,(xn,yn)]
        color: color del punto (BGR)
        radius: radio del punto a dibujar
    Returns:
        imagen con los puntos dibujados
    """
    image = cv2.circle(image, tuple(points), radius, color, -1)
    return image

def random_crop_binary_model(image_path, crop_size, num_crops, save_folder, num):
    
    """
    Toma recortes aleatorios de una imagen de un tamaño especificado y devuelve un array de numpy con las imágenes
    recortadas. Opcionalmente, también guarda las imágenes recortadas en una carpeta especificada.

    Args:
        image_path (str): Ruta de la imagen.
        crop_size (int): Tamaño de los recortes cuadrados a tomar.
        num_crops (int): Número de recortes aleatorios a tomar.
        save_folder (str, optional): Ruta de la carpeta donde guardar los recortes. Por defecto es None (no se guarda).

    Returns:
        numpy.ndarray: Array de numpy con los recortes aleatorios de la imagen de tamaño (num_crops, crop_size, crop_size, 3)
    """
    
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image_gray.shape

    for i in range(num_crops):
        # Genera coordenadas aleatorias para el recorte
        x = np.random.randint(0, width - crop_size + 1)
        y = np.random.randint(0, height - crop_size + 1)

        # Toma el recorte
        crop = image_gray[y:y+crop_size, x:x+crop_size]

        # Guarda la imagen si se especificó una carpeta
        filepath = save_folder + str(num) +"-"+ str(i)+ '.jpeg'
        cv2.imwrite(filepath, crop)

def clustering(candidate_center,threshold_dis):
    x = candidate_center[:,1]
    y = candidate_center[:,0]
    group_distance = []
    for i in range(len(candidate_center)):
        xpoint, ypoint = x[i], y[i]
        xTemp, yTemp = x, y 
        distance = np.sqrt(pow((xpoint-xTemp),2)+pow((ypoint-yTemp),2))
        distance_matrix = np.vstack((np.array(range(len(candidate_center))),distance))
        distance_matrix = np.transpose(distance_matrix)
        distance_sort = distance_matrix[distance_matrix[:,1].argsort()] 
        distance_sort = np.delete(distance_sort,0,axis = 0)
        thre_matrix = distance_sort[distance_sort[:,1]<=threshold_dis]
        thre_point = thre_matrix[:,0]
        thre_point = thre_point.astype(int)
        thre_point = thre_point.tolist()
        thre_point.insert(0,i)
        group_distance.append(thre_point)
    
    group_clustering = [[]] 
    
    for i in range(len(candidate_center)):
        m1 = group_distance[i]
        for j in range(len(group_clustering)):
            m2 = group_clustering[j]
            com = set(m2).intersection(set(m1))
            if len(com) == 0:
                if j == len(group_clustering)-1:
                    group_clustering.append(m1)
            else:
                m = set(m1).union(set(m2))
                group_clustering[j] = []
                group_clustering[j] = list(m)
                break
    group_clustering.pop(0)
    return group_clustering  #the group of candiate center

def center_clustering(candidate_center,group_clustering):
    final_result = []
    for i in range(len(group_clustering)): 
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:,1] 
        yz = points_coord[:,0]
        x_mean = np.mean(xz)
        y_mean = np.mean(yz)
        final_result.append([y_mean,x_mean])
    final_result = np.array(final_result)
    final_result = final_result.astype(int)
    return final_result

def ImgDraw_prediction(patch_img, parameter=0.5):
    
    """
    Predicción de barra.

    ---------------------------------------------------------------------------------------------
    """
    folder_path_img = patch_img #"G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-36.jpeg"
    RL = load_model("RL.h5")
    BC = load_model("BC.h5")
    
    img_pred_Fr = random_crops_RL(folder_path_img)
    
    pred_RL = []
    for i in range(5):
        img_pred_Fr_A = img_pred_Fr[i,:,:]
        img_pred_Fr_A = np.expand_dims(img_pred_Fr_A, axis=0)
        pred_RL.append(RL.predict(img_pred_Fr_A))
    
    Pred_RL_Arr = np.array(pred_RL)
    promedio_RL_Arr = np.mean(Pred_RL_Arr)
    
    Fr = (64 / promedio_RL_Arr)*0.95
    
    img = cv2.imread(folder_path_img, cv2.IMREAD_GRAYSCALE)
    
    h, w = img.shape
    
    h_fr = h*Fr
    
    w_fr = w*Fr
    
    img_resc =  cv2.resize(img, (int(h_fr), int(w_fr)))  
    
    Arr_pred, coord = image_to_array_crop64(img_resc)
    
    size_im, _, _= Arr_pred.shape
    
    BC_pred = BC.predict(Arr_pred)
    g_coord = []
    
    for i in range(size_im):
        
        if BC_pred[i] > parameter:
            g_coord.append(coord[i,:])
            #img_array = (Arr_pred[i] * 255).astype(np.uint8)
    
            #img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            #cv2.imshow("Image", img_cv2)
            #cv2.waitKey(0)
    grup_cluster=clustering(np.array(g_coord),20)
    final_coord = center_clustering(np.array(g_coord),grup_cluster)
    for i in range(len(final_coord)):
        img_out = draw_points(img_resc, ([final_coord[i,1], final_coord[i,0]]))
    
    
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    return grup_cluster


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_image_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_image_button.pack(side="top")

        self.process_image_button = tk.Button(self, text="Process Image", command=self.process_image, state="disabled")
        self.process_image_button.pack(side="top")

        self.quit_button = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit_button.pack(side="bottom")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.process_image_button["state"] = "normal"

    def process_image(self):
        #image = cv2.imread(self.image_path)
        #print(self.image_path)
        # Aquí es donde puede llamar a su función de procesamiento de imagen
        ImgDraw_prediction(self.image_path, 0.8)



#root = tk.Tk()
#app = Application(master=root)
#app.mainloop()


preProcessing('G:\Maestria\Analisis numerico\Proyecto\Test\img-', 8, 64)

#patch_img =  "G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-2.jpeg"

#A=ImgDraw_prediction(patch_img, parameter=0.5)

"""
Pruebas de filtros 

---------------------------------------------------------------------------------------------
img1 = cv2.imread('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-7.jpeg',cv2.COLOR_BGR2GRAY)
filtro_media = cv2.medianBlur(img1, 5)
canny = cv2.Canny(filtro_media, 50, 100)
(contornos, jerarquia) = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours (img1, contornos, -1, (0,0,255), 2)
#ret1, th1 = cv2.threshold(img1,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.subplot(1,2,1)
plt.imshow(img1,'gray',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.imshow(canny,'gray',vmin=0,vmax=255)
plt.show()
cv2.imwrite('G:\Maestria\Analisis numerico\Proyecto\Imagens\IMGbinary.jpeg', img1)
"""

"""

Entrenar modelo Binary
--------------------------------------------------------------------------------------------


folder_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\BC\img-"
X_train = load_images_x_train(folder_path, 1256)

Y_train = load_csv('G:\Maestria\Analisis numerico\Proyecto\Imagens\BC\BinaryC.csv',1)


BC = create_model_BC((64,64, 1))

BC = train_model_CNN(BC, X_train, Y_train, 10000, 33)

BC.save('BC.h5')
"""


"""

Imagenes Fr para entrenamiento

--------------------------------------------------------------------------------------------
folder_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\Img-"
save_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\Img-"

for i in range(1,55):
    
    A = random_crops_RL(folder_path+str(i)+".jpeg", save_path+str(i)+"-")
"""    

"""
Entrenamiento del FR

--------------------------------------------------------------------------------------------

folder_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\img-"
X_train = load_images_x_train(folder_path,271)

Y_train = load_csv('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\FrLabel.csv',1)

Y_train = np.repeat(Y_train, 5)

RL = create_model_RL((64,64, 1))

RL = train_model_CNN(RL, X_train, Y_train, 200, 5)

RL.save('RL.h5')
"""


# random_crop_binary_model('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-1.jpeg', 53, 20, 'G:\Maestria\Analisis numerico\Proyecto\Imagens\BC\img-')


"""
Crear puntos en imagenes

--------------------------------------------------------------------------------------------
image = cv2.imread('G:\Maestria\Analisis numerico\Proyecto\Imagens\BC\img-0.jpeg')
points = [(10, 20), (30, 40)]
image_with_points = draw_points(image, points)
cv2.imwrite('mi_imagen_con_puntos.png', image_with_points)
"""


"""
#Crear dataset para entrenamiento binaryC

#---------------------------------------------------------------------------------------------
folder_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling"
save_path = "G:\Maestria\Analisis numerico\Proyecto\Imagens\BC\img-"

Fr_real = load_csv("G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\FrLabel.csv", 1)

i = 0
for filename in os.listdir(folder_path):
    if filename.startswith("img-") and filename.endswith(".jpeg"):
        img_path = os.path.join(folder_path, filename)
        random_crop_binary_model(img_path, Fr_real[i], 20, save_path, i)
        i= i+1
   
"""

"""
Realizar la predicción

----------------------------------------------------------------------------------------------

RL = load_model('RL.h5')
img = cv2.imread('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgFr\img-10.jpeg', cv2.IMREAD_GRAYSCALE)

# Normalizar la imagen
img_norm = img / 255.0

# Agregar una dimensión adicional para representar el batch_size
img_norm = np.expand_dims(img_norm, axis=0)

# Hacer la predicción
pred = RL.predict(img_norm)

image_array, coord = image_to_array('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-1.jpeg')

A = random_crops('G:\Maestria\Analisis numerico\Proyecto\Imagens\ImgScaling\img-1.jpeg')

"""












































