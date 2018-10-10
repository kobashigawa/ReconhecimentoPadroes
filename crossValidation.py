import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
#https://mclguide.readthedocs.io/en/latest/sklearn/cv.html

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
    header=None, 
    sep=',')

df.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']
df.dropna(how="all", inplace=True)

df2=df.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

atributos=df2[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]
classe=df2[['classe']]
X, y = atributos.values, classe.values

cv = StratifiedKFold(n_splits=3, shuffle=True)
cv.get_n_splits(X, y)

print(classe.size)
print(cv)

for train_index, test_index in cv.split(X, y):
	#print("TRAIN:", train_index, "TEST:", test_index)
	print("%s %s" % (train_index.size,test_index.size))
