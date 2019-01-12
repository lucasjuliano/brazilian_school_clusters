# Dependencies
import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#Função recebe número de clusters e retorna um pandas com os dados tagueados de acordo com o cluster e valores dos centroides
def cluster(num_clusters):

    # Dicionário de codigos de UFs
    co_ufs = {
                11 : "Norte",
                12 : "Norte",
                13 : "Norte",
                15 : "Norte",
                14 : "Norte",
                16 : "Norte",
                17 : "Norte",
                21 : "Nordeste",
                22 : "Nordeste",
                23 : "Nordeste",
                24 : "Nordeste",
                25 : "Nordeste",
                26 : "Nordeste",
                27 : "Nordeste",
                28 : "Nordeste",
                29 : "Nordeste",
                31 : "Sudeste",
                32 : "Sudeste",
                35 : "Sudeste",
                33 : "Sudeste",
                41 : "Sul",
                42 : "Sul",
                43 : "Sul",
                50 : "Centro-oeste",
                51 : "Centro-oeste",
                52 : "Centro-oeste",
                53 : "Centro-oeste",
            }

    # Dicionário de tipos de escola
    tp_escola = {
                1 : "Federal",
                2 : "Estadual",
                3 : "Municipal",
                4 : "Privada"
    }

    #Dimensões a serem consideradas durante clustering
    dimensions = ['IN_LABORATORIO_INFORMATICA','IN_LABORATORIO_CIENCIAS','IN_QUADRA_ESPORTES_COBERTA','IN_QUADRA_ESPORTES_DESCOBERTA','IN_BIBLIOTECA','IN_INTERNET','IN_ALIMENTACAO']

    # Puxar CSV
    data_schools = pd.read_csv("cluster_data.csv",sep=',')

    # Retirar escolas que não estão em funcionamento
    data_schools = data_schools[data_schools['TP_SITUACAO_FUNCIONAMENTO'] == 1]

    # Retirar linhas com NaN
    data_schools = data_schools.dropna()

    # Colocar nome dos estados
    data_schools['Re'] = data_schools['CO_UF'].map(co_ufs)

    # Transformar categorias 1 e 2 do IN_LOCALIZACAO
    f = lambda x: 0 if x==2 else 1
    data_schools['IN_URBANO'] = data_schools['IN_URBANO'].map(f)

    # Transformar pandas em arrays
    data = data_schools[dimensions].iloc[:,0:len(dimensions)].values

    #Normalizar os dados
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Rodar fit
    kmeans = KMeans(n_clusters = num_clusters, max_iter = 1000) # prop -> init = 'random'
    kmeans.fit(data_scaled)

    #Adicionando labels ao conjunto original
    data_schools['cluster_label'] = pd.Series(kmeans.labels_).values

    return data_schools,kmeans
