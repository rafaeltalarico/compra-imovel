#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#tipo de renda: 1- empresário; 2- autônomo; 3- assalariado
# possui imóvel: 1- não; 2- sim
#comprou: 0- não; 1-sim

tabela = pd.read_excel("BaseDados_RegressaoLogistica.xlsx")
display(tabela)


# In[2]:


#sns.set para plotar todos os gráficos de uma vez
# 'figure.figsize': tamanho da plotagem
sns.set(font_scale=1.3, rc={'figure.figsize':(15,10)})

#bins: largura da barra; quanto maior o "bins", menor a largura
eixo = tabela.hist( bins=20, color='blue')


# In[3]:


plt.figure(figsize=(10,5))
sns.boxplot(data=tabela, x='Tipo Renda', y='Renda')


# In[4]:


plt.figure(figsize=(10,5))
sns.boxplot(data=tabela, x='Possui Imóvel', y='Renda')


# In[5]:


# outliers: pontos fora da curva/amostragem
plt.figure(figsize=(10,5))
sns.boxplot(data=tabela, x='Comprou?', y='Renda')


# In[6]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=tabela, x='Renda', y='Comprou?')


# In[7]:


caracteristica = tabela.iloc[:,1:4].values
previsor = tabela.iloc[:,4:5].values
display(caracteristica)
display(previsor)


# In[8]:


from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(caracteristica, previsor)
print(len(x_treino))
print(len(x_teste))


# In[9]:


from sklearn.linear_model import LogisticRegression
funcao_logistica = LogisticRegression()
funcao_logistica.fit(x_treino, y_treino)
print(funcao_logistica)


# In[10]:


#predict: função para prever
previsoes = funcao_logistica.predict(x_teste)
display(previsoes)


# In[11]:


y_teste


# In[12]:


#assertividade
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_teste, previsoes))


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_teste, previsoes))


# In[14]:


from sklearn.metrics import r2_score
print(r2_score(y_teste, previsoes))


# In[15]:


#novo cliente
salario = 11500
Tipo_Renda = 1
Possui_Imovel = 0

parametro = [[salario, Tipo_Renda, Possui_Imovel]]
fazendo_previsao = funcao_logistica.predict(parametro)
probabilidade = funcao_logistica.predict_proba(parametro)

if fazendo_previsao == 0:
    print('não irá comprar')
    print(probabilidade)
else:
    print('vai comprar')
    print(probabilidade)


# In[ ]:




