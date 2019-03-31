# Dependencies
import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from cluster import *
import csv

# Ver relaçao entre aumento de clusters e diminuicao do erro
print('Clusters - Erro')

#Dicionário dos resultados
dict_results = {
                    "Clusters" : "Erro"
                }

#Setando erro passado
past_error = 0

for i in range(1, 9):

    #Inicializando o algoritmo de Kmeans
    data_schools,kmeans,dimensions = cluster(i)

    #Mostrando erro quadrado
    if(past_error == 0):
        print(i,np.round(kmeans.inertia_,decimals=0),past_error)
    else:
        #Porcentagem do erro é calculada
        past_error = np.round((1 - kmeans.inertia_/past_error)*100,decimals=0)
        print(i,np.round(kmeans.inertia_,decimals=0),past_error)


    #Adiciona ao Dicionário
    dict_results[i] = past_error

    #Valor do último erro é colocado aqui
    past_error = np.round(kmeans.inertia_,decimals=0)

#Salva como csv
with open('cluster_evaluations.csv','w') as f:
    w = csv.writer(f)
    w.writerows(dict_results.items())
