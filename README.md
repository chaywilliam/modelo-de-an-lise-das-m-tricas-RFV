#  Repositório de Análise Exploratória de Dados de Clientes
Este repositório contém um script Python para análise exploratória de dados de clientes, com foco na análise RFM (Recency, Frequency, Monetary) e segmentação de clientes por meio de clustering com KMeans. O objetivo é identificar diferentes segmentos de clientes com base em seu comportamento de compra e fornecer insights e recomendações para ações de marketing direcionadas.

## Descrição do Projeto
O arquivo data.csv contém dados de transações de clientes, incluindo informações sobre data da fatura, número da fatura, ID do cliente, quantidade, preço unitário e país. O script Python realiza as seguintes etapas:

## 1.Carregamento e Análise Exploratória Inicial: Carrega os dados, exibe as primeiras linhas, informações sobre os tipos de dados, resumo estatístico e verifica valores nulos.
## 2.Análise das Métricas RFV: Calcula as métricas RFV (Recência, Frequência e Valor Monetário) para cada cliente.
## 3.Pré-processamento dos Dados: Normaliza os dados usando MinMaxScaler para preparar os dados para a análise de clustering.
## 4.Seleção de Features: Calcula a matriz de correlação para identificar relações entre as variáveis.
## 5.Escolha do Algoritmo de Clustering: Utiliza o algoritmo KMeans para agrupar os clientes em diferentes segmentos.
## 6.Determinação do Número Ótimo de Clusters (K): Utiliza o Método do Cotovelo e o Silhouette Score para determinar o número ideal de clusters.
## 7.Análise dos Clusters: Analisa as características de cada cluster, como estatísticas descritivas, boxplots e histogramas.
## 8.Insights e Recomendações: Fornece insights e recomendações com base nos clusters identificados, como programas de fidelidade, campanhas de marketing direcionadas e gerenciamento de estoque.

## Estrutura do Repositório
.
├── README.md
└── customer_segmentation.ipynb

## Como Executar o Código
### 3.Clone este repositório:

Bash

git clone https://github.com/seu-usuario/nome-do-repositorio.git

### 2.Instale as dependências:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn

### 3.Execute o notebook Jupyter:

Bash

jupyter notebook customer_segmentation.ipynb

## Carregamento e Análise Exploratória Inicial
O script começa importando as bibliotecas necessárias e carregando os dados do arquivo CSV usando a biblioteca Pandas:

Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/data.csv', encoding='CP1252')
Em seguida, são exibidas as primeiras linhas do DataFrame usando df.head(), informações sobre os tipos de dados usando df.info(), um resumo estatístico usando df.describe() e a quantidade de valores nulos em cada coluna usando df.isnull().sum().

## Análise das Métricas RFV
As métricas RFV (Recência, Frequência e Valor Monetário) são calculadas para cada cliente. A Recência é calculada como o número de dias desde a última compra, a Frequência é o número de compras e o Valor Monetário é a soma do valor total das compras.

Python

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
if 'TotalPrice' not in df.columns:
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': lambda x: len(x),
    'TotalPrice': lambda x: x.sum()
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

## Pré-processamento dos Dados
Os dados são normalizados usando MinMaxScaler para preparar os dados para a análise de clustering.

Python

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_features = ['Recency', 'Frequency', 'Monetary']
rfm_scaled = scaler.fit_transform(rfm[numeric_features])

## Seleção de Features
A matriz de correlação é calculada para identificar relações entre as variáveis.

Python

correlation_matrix = rfm.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

## Escolha do Algoritmo de Clustering
O algoritmo KMeans é utilizado para agrupar os clientes em diferentes segmentos.

Python

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rfm_scaled)
labels = kmeans.labels_
rfm['Cluster'] = labels

## Determinação do Número Ótimo de Clusters (K)
O Método do Cotovelo e o Silhouette Score são utilizados para determinar o número ideal de clusters.

Python

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(rfm_scaled, labels)
print("Silhouette score:", silhouette_avg)

## Análise dos Clusters
As características de cada cluster são analisadas, como estatísticas descritivas, boxplots e histogramas.

Python

rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median', 'sum']
})

sns.boxplot(x='Cluster', y='Monetary', data=rfm)
plt.title('Comparação do Valor de Compra por Cluster')
plt.show()

def plot_histograms(df, numeric_columns, cluster_column='Cluster'):
    for column in numeric_columns:
        sns.histplot(data=df, x=column, hue=cluster_column, kde=True)
        plt.title(f'Distribuição de {column} por Cluster')
        plt.show()

numeric_columns = ['Recency', 'Frequency', 'Monetary']
plot_histograms(rfm, numeric_columns)

cluster_counts = rfm['Cluster'].value_counts()
sns.countplot(x='Cluster', data=rfm)
plt.title('Distribuição dos Clientes por Cluster')
plt.ylabel('Número de Clientes')
plt.show()

## Insights e Recomendações
Com base nos clusters identificados, são fornecidos insights e recomendações para ações de marketing direcionadas. Por exemplo:

### Cluster 0 (Os Fiéis): Clientes com alta frequência de compra e alto valor médio por pedido.
Recomendação: Criar programas de fidelidade, oferecer produtos personalizados e enviar comunicação exclusiva.

### Cluster 1 (Os Ocasionais): Clientes com baixa frequência de compra e valor médio por pedido moderado.
Recomendação: Criar campanhas de reaproximação, analisar carrinhos abandonados e sugerir produtos complementares.

### Cluster 2 (Os Exploradores): Clientes com alta frequência de compra, mas baixo valor médio por pedido.
Recomendação: Destacar produtos com maior margem de lucro, criar kits e pacotes e implementar programas de referência.

# Próximos Passos
### Explorar outros algoritmos de clustering, como DBSCAN ou Hierarchical Clustering.
### Realizar uma análise mais aprofundada dos dados para identificar outros padrões e insights relevantes.
### Utilizar os clusters para criar campanhas de marketing mais direcionadas e personalizadas.

Este README fornece uma visão geral do projeto e como executar o código. Sinta-se à vontade para explorar o notebook Jupyter para obter mais detalhes sobre a implementação e os resultados da análise.
