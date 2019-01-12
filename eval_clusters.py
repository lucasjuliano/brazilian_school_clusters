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

# Ver relaçao entre aumento de clusters e diminuicao do erro
print('Clusters - Erro')
for i in range(1, 15):
    #Inicializando o algoritmo de Kmeans
    data_schools,kmeans = cluster(i)

    #Mostrando erro quadrado
    print(i,np.round(kmeans.inertia_,decimals=0))
