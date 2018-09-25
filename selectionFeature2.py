import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

df = pd.read_csv(
	filepath_or_buffer='C:/Users/Luiz/teste/adult.csv',
	header=None,
	sep=',')

df.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']
df.dropna(how="all", inplace=True)
# drops the empty line at file-end

df2=df.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

atributos=df2[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]
classe=df2[['classe']]

X, y = atributos.values, classe.values
##print(X.shape)

selector = SelectKBest(f_classif, k=4)
X_new = selector.fit_transform(X, y)

#print(X_new.shape)

mask = selector.get_support()
new_features = atributos.columns[mask]

print(new_features)