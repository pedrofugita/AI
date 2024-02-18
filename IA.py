# IMPORTA BASE DE DADOS
import pandas as pd
tabela = pd.read_csv("clientes.csv")
print("BASE DE DADOS:")
print(tabela)
print(tabela.info())    # verifica se existem valores vazios



# PREPARAR BASE DE DADOS (PRÉ-PROCESSAMENTO)

# Label Encoder (associa um texto a um número para utilizar somente números)
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])
print("\nBASE DE DADOS COM TEXTO TRANSFORMADO:")
print(tabela.info())

# caso houverem muitas colunas, utilizar um loop
#for coluna in tabela.columns:
#    if tabela[coluna].dtype == "object" and coluna != "score_credito":
#        tabela[coluna] = codificador.fit_transform(tabela[coluna])

# define com quais colunas da base de dados será realizada o treinamento
y = tabela["score_credito"] # y é o que se quer encontrar
x = tabela.drop(columns=["score_credito", "id_cliente"])   # x são valores que influenciam no cálculo e .drop elimina os valores que não influenciam



# TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=1) # 30% da base de dados será destinada aos testes



# CRIAÇÃO DO MODELO DE IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier() # modelo de árvore de decisão (faz perguntas para a base de dados até afunilar)
modelo_knn = KNeighborsClassifier() # modelo de vizinhos próximos (método gráfico que cria divisórias de classificação)

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
print("TREINAMENTO COMPLETO")



# ESCOLHER O MELHOR MODELO
from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())
print("\nACURÁCIA DOS MÉTODOS:")
print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsao_knn))



# FAZER NOVAS PREVISÕES
novos_clientes = pd.read_csv("novos_clientes.csv")
print("\nDADOS DOS NOVOS CLIENTES:")
print(novos_clientes)
novos_clientes["profissao"] = codificador.fit_transform(novos_clientes["profissao"])
novos_clientes["mix_credito"] = codificador.fit_transform(novos_clientes["mix_credito"])
novos_clientes["comportamento_pagamento"] = codificador.fit_transform(novos_clientes["comportamento_pagamento"])

previsoes = modelo_arvoredecisao.predict(novos_clientes)
print("\nPREVISÃO DE SCORE DOS NOVOS CLIENTES:")
print(previsoes)
