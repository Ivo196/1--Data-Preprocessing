# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:00:54 2023

@author: ivoto
"""

#Con control enter lo que hago es cargar esa linea 
#Plantilla de Pre-Procesado 
#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 

dataset = pd.read_csv('Data.csv')
#Definimos variable independientes, [filas, columnas (:-1 por que quiero todas menos la ultima)]
X = dataset.iloc[:, :-1].values 
#Definimos y(minuscula ya que es un vector en vez de una matriz) variable dependiente 
y = dataset.iloc[:, 3].values 

#Codificamos datos categoricos

#Codificar(pasar a numeros) datos categoricos(Es decir de categorias)
#Cabiamos por ej los pais a un numero correspondiente, como France = 1, Spain = 2 
#Y asi hasta tener todas las categorias ya codificadas
#Usamos la libreria sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Creamo dos, uno para los paises y otro para la matriz de caracteristicas(y)
labelencoder_X = LabelEncoder() #El constructor no necesita nada 
labelencoder_X.fit_transform(X[:,0])  #Me devuelve un array, de la codifcicaion 
#Ahora le doy el valor del encouder 
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#Ahora ese valor es categorico por lo que tenemos que trasnformar a variable Dummy(Dummyficacion)
#Lo hacemos utilizando la misma libreria de skleanr y la transformacion que hicimos en el paso anterior
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)
#Ahora hacemos la tranformacion de la matriz de carac. pero como es yes o no, es sencillo, ya que seria 1 o 0
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#Eso es todo para tratar con variables categoricas y ordinales (SIEMPRE que tenga un orden tipo XL, L, M es Ordinal)