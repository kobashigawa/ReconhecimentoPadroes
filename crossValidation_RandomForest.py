import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from numba import jit

from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv(
    filepath_or_buffer='adultinho.data',
    header=None, 
    sep=',')

dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

dataset.dropna(how="all", inplace=True) # drops the empty line at file-end

somente_numericos=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])
somente_numericos = somente_numericos.sort_values(by=['classe'])

features_full=somente_numericos[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]

#Requerido: package numba - pip install numba --user
#Kobashi: Apenas os do PCA = sem Idade e Peso final
#features_full=somente_numericos[['Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]

# Relief - sem peso final
#features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Ganhos de Capital', 'Horas de trabalho por semana']]

# k best - sem perdas de capital 
#features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Ganhos de Capital', 'Horas de trabalho por semana']]

#features_full=somente_numericos[['Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]
#Relief - 'Idade', 'Anos de Estudo', 'Horas de trabalho por semana', 'Ganhos de Capital', 'Peso final', 'Perdas de Capital'
#features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Horas de trabalho por semana', 'Ganhos de Capital']]
#SelectKBest - 'Anos de Estudo', 'Idade', 'Horas de trabalho por semana', 'Ganhos de Capital', 'Perdas de Capital', 'Peso final'
#features_full=somente_numericos[['Anos de Estudo', 'Idade', 'Horas de trabalho por semana', 'Ganhos de Capital']]

features=np.array(features_full.values, dtype=np.float64)
#features=np.array(dataset.values, dtype=np.float64)

labels=somente_numericos[['classe']].values

# Necessario usar 0 e 1 para recall e precision
# https://stackoverflow.com/questions/39187875/scikit-learn-script-giving-vastly-different-results-than-the-tutorial-and-gives
# Ha 3 formas descritas no link
labels_somente_numericos = [1 if i == ' <=50K' else 0 for i in labels]

# Cross validation n splits = 3
cvn3 = StratifiedKFold(n_splits=3, shuffle=True)
cvn3.get_n_splits(features, labels_somente_numericos)

# Cross validation n splits = 5
cvn5 = StratifiedKFold(n_splits=5, shuffle=True)
cvn5.get_n_splits(features, labels_somente_numericos)


models = []

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

models.append(('RndForest mtry=3 e ntree=500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=3 e ntree=1000', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=3 e ntree=1500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))

models.append(('RndForest mtry=4 e ntree=500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=4 e ntree=1000', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=4 e ntree=1500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))

models.append(('RndForest mtry=5 e ntree=500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=5 e ntree=1000', RandomForestClassifier(min_samples_split=4, n_estimators=500)))
models.append(('RndForest mtry=5 e ntree=1500', RandomForestClassifier(min_samples_split=4, n_estimators=500)))


# Parametros do scoring http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print("Iniciando cross validation MLP")


for name, model in models:
    print("Model:{0} and n splits = 3".format(name))
    score_acc = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='accuracy')
    print("Accuracy:\n%0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='precision')
    print("Precision:\n%0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='recall')
    print("Recall:\n%0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))

    print("Model:{0} and n splits = 5".format(name))
    score_acc = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='accuracy')
    print("Accuracy:\n%0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='precision')
    print("Precision:\n%0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='recall')
    print("Recall:\n%0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))
	

