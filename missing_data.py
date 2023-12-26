# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:01:51 2023

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

#Resolvemos el problema de datos faltantes o desconocidos NaN  
from sklearn.impute import SimpleImputer 
#missing_values = es el valor que queremos remplazar
#strategy = es el tipo de estrategia, media, mediana
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
#ahora se lo tenemos que pasar al matriz de caracteristicas 
#siempre usamos fit y le pasamos la matriz X que es la quetiene el error
imputer = imputer.fit(X[:,1:3]) #Tomamos de la una a la 3 ya que si tomo a la 2 no es tomado por Py
X[:, 1:3] = imputer.transform(X[:,1:3]) #Le Pasamos la nueva matriz a la orginal y ya esta, lo podemos ver por consola
