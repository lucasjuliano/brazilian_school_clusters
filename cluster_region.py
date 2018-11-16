# Dependencies
import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

#Numero de Clusters
num_clusters = 4

#Dimensões a serem consideradas durante clustering
dimensions = ['IN_LABORATORIO_INFORMATICA','IN_LABORATORIO_CIENCIAS','IN_QUADRA_ESPORTES','IN_BIBLIOTECA','IN_INTERNET','IN_ALIMENTACAO']

# Constantes de de interpretação
pos_lim = 0.80
quasi_pos_lim = 0.60
quasi_neg_lim = 0.40
neg_lim = 0.20

# Interpreta os dados transformando porcentagem em informacao
def interpreter(num):
    if (num >= pos_lim):
        return "POSSUI"
    elif (pos_lim > num > quasi_pos_lim):
        return "PROVAVELMENTE POSSUI"
    elif (num <= neg_lim):
        return "NÃO POSSUI"
    elif (quasi_neg_lim < num < neg_lim):
        return "PROVAVELMENTE NAO POSSUI"
    else:
        return "PODE TER OU NÃO"

# Pega array e transforma em dicionario com porcentagem dos valores
def percentagearray(array):
    num_total = len(array)
    dict = collections.Counter(array)
    for key,val in dict.items():
        dict[key] = round(dict[key]/num_total * 100, 0)
    return dict;

# Puxar CSV
data_schools = pd.read_csv("trial_data.csv",sep=',')

# print("******** Data Schools  ********")

# Retirar linhas com NaN
data_schools = data_schools.dropna()

# Colocar nome dos estados
data_schools['Re'] = data_schools['CO_UF'].map(co_ufs)

# print(data_schools[['UF','CO_UF','IN_LABORATORIO_INFORMATICA','IN_LABORATORIO_CIENCIAS','IN_BIBLIOTECA']].groupby(['UF'], as_index=False).mean().sort_values(by='IN_LABORATORIO_INFORMATICA', ascending=False))

# Transformar em arrays
data = data_schools[dimensions].iloc[:,0:6].values

# Ver relaçao entre aumento de clusters e diminuicao do erro
# print('Clusters - Erro')
# for i in range(1, 15):
    # Inicializando o algoritmo de Kmeans
    # kmeans = KMeans(n_clusters = i, init = 'random')
    # kmeans.fit(data)

    # Mostrando erro quadrado
    # print(i,kmeans.inertia_)

# Rodar fit
kmeans = KMeans(n_clusters = num_clusters, max_iter = 1000) # prop -> init = 'random'
kmeans.fit(data)

#Adicionando labels ao conjunto original
data_schools['cluster_label'] = pd.Series(kmeans.labels_).values

# Dispersão geral entre os clusters
print('Distribuição entre os clusters geral')
print(percentagearray(kmeans.labels_))

# Print ver centroides
print('Descrição dos perfis')

# Pega valores de centroides
values = np.around(kmeans.cluster_centers_,decimals=2)
# print(values)

# Roda por cada perfil para mostrar as dimensoes deles
for i in range(0, num_clusters):
    print("Cluster ",i)
    for d in range(0,len(dimensions)):
        print(dimensions[d],interpreter(values[i][d]))
    print("--")

print("***************************************************************")

print("-----Por tipo de escola:")
for item in np.unique(np.array(data_schools['TP_DEPENDENCIA'].values)):
    # Filtrando para um tipo de escola
    data_tp = data_schools[data_schools['TP_DEPENDENCIA'] == item]
    print(tp_escola[item])
    print(percentagearray(data_tp['cluster_label']))
    print("***************************************************************")

print("-----Por regiões:")

for item in np.unique(np.array(data_schools['Re'].values)):
    # Filtrando para uma regiao específico
    data_re = data_schools[data_schools['Re'] == item]
    print(item)
    print(percentagearray(data_re['cluster_label']))
    print("***************************************************************")


# Porcentagem total de valores vazios
# print(data_schools['IN_ALIMENTACAO'].notnull().mean())

# Porcentagem por estado de valores vazios
# print(data_schools.groupby(['CO_UF'], as_index=False).apply(lambda x: x.notnull().mean()))

# print(data_schools.info())
# print(data_schools.describe())
# print(data_schools.columns.values)

# print("Missed itens")
# print(data_schools.isna().sum())
#
# print("Juntando por estados")
# print(data_schools.groupby(['CO_UF'], as_index=False).mean())

# print(data_schools[["Survived","Pclass"]].groupby(['Survived'],as_index=False).mean().sort_values(by='Pclass', ascending=False))
# print("/n")

# g = sns.FacetGrid(data_schools, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
