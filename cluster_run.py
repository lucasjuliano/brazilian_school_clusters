# Dependencies
import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from cluster import cluster

#Numero de Clusters
num_clusters = 3

# Constantes de interpretação
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

def print_results_by(message,row):
    # Função para qual a distribuição dos clusters comparando com algum critério - regiao, publico/privado, rural/urbano
    # Também salva os valores em um dataframe e depois em um .csv
    print("-----",message,":")
    print("\n")
    df = pd.DataFrame()

    for item in np.unique(np.array(data_schools[row].values)):
        # Filtrando para uma regiao específico
        data_re = data_schools[data_schools[row] == item]
        percentage_by_criteria = percentagearray(data_re['cluster_label'])
        print(item, percentage_by_criteria)
        print("\n")

        #Cada vez que roda, adiciona o dicionário ao dataframe
        percentage_by_criteria['Nome'] = item
        df = df.append(percentage_by_criteria, ignore_index=True)

    # Salvando os valores desse DataFrame
    df.to_csv(row + '_values.csv')

# Manda número de clusters e recebe os dados do fit e o dataframe
data_schools,kmeans,dimensions = cluster(num_clusters)


# Dispersão geral entre os clusters
print('Distribuição entre os clusters geral')
print(percentagearray(data_schools['cluster_label'].tolist()))

# Print ver centroides
print('Descrição dos perfis')

# Pega valores de centroides
centroides = np.around(kmeans.cluster_centers_,decimals=2)
#print(centroides)

print('Quantas escolas estão sendo consideradas nesse (rows,cols):',data_schools.shape)

# Roda por cada cluster para mostrar o valor das dimensoes deles
for i in range(0, num_clusters):
    print("Cluster ",i)
    for d in range(0,len(dimensions)):
        print(dimensions[d],'-',interpreter(centroides[i][d]))
        #print(dimensions[d],'-',centroides[i][d])
    print("--")

print("***************************************************************")

# Mostrar resultados por tipo escola publico ou privado
print_results_by('Por tipo de escola','TP_DEPENDENCIA')

# Mostrar por regiões do brasil
print_results_by('Por regiões','CO_REGIAO')

# Mostrar por rural ou urbano
print_results_by('Por rural (0) e urbano (1)','TP_LOCALIZACAO')

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
