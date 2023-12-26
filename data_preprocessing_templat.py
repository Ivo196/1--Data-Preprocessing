# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:28:21 2023

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





#Dividir el dataset en conjunto de entrenamiento y de testing
#Utilizamos un libreria sklearn, model_selection (muy utilizidad para cross-validation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )

#Escalado de variables

#Como la distancia euclidea (pitagoras de las variables ) puede tomar datos muy grande de una de las variables, lo que se hace es normalizar 
#Esto se hace para que un variable de valor muy grande no domine sobre el resto
#Escalar los datos = Normalizar los datos (Escalar a 0 y -1 correspode que el val max es 1 y el valor min es -1 ) 
#Hay dos metodos Standardisation(Campara de Gauss) y Normalisation(0 a 1 Lineal)

from sklearn.preprocessing import StandardScaler
#Escalamos el conjunto de entrenamiento 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
#El conjunto de test tiene que escalar con la misma tranformacion, no podemos usar una distinta para el conj de test 
X_test = sc_X.transform(X_test) #Solo detecta la transformacion y la aplica
#Ahora las variables de y-train e y_test no lo hacemos ya que es de clasificaion, por lo que no normalizamos 
#Si utilizaramos un algoritmo de prediccion(regresion lineal) hay que normalizar la y_train






























