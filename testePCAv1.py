import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
    header=None, 
    sep=',')

df.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

df.dropna(how="all", inplace=True) # drops the empty line at file-end

df2=df.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

atributos=df2[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]
classe=df2[['classe']]

x_std = StandardScaler().fit_transform(atributos)

# features are columns from x_std
features = x_std.T 
covariance_matrix = np.cov(features)
print(covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)

# We reduce dimension to 1 dimension, since 1 eigenvector has 73% (enough) variances
for x in range(0,6):
	print  (eig_vals[x] / sum(eig_vals))
	
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print (var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 6))
    plt.bar(range(6), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(6), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()