import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from numba import jit

from sklearn.neural_network import MLPClassifier

dataset = pd.read_csv(
    filepath_or_buffer='adult.data',
    header=None, 
    sep=',')

dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

dataset.dropna(how="all", inplace=True) # drops the empty line at file-end

somente_numericos=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

somente_numericos = somente_numericos.sort_values(by=['classe'])

#features_full=somente_numericos[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]

#Kobashi: Apenas os do PCA = sem Idade e Peso final
#features_full=somente_numericos[['Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]

#Kobashi: Relief - sem peso final
features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Ganhos de Capital', 'Horas de trabalho por semana']]

#Kobashi: k best - sem perdas de capital 
features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Ganhos de Capital', 'Horas de trabalho por semana']]

features=np.array(features_full.values, dtype=np.float64)

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

# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

models.append(('MLP  2 neuronios, a taxa 0,1 ', MLPClassifier(hidden_layer_sizes=(2, ),  learning_rate_init=0.1)))
models.append(('MLP  2 neuronios, a taxa 0,05 ', MLPClassifier(hidden_layer_sizes=(2, ),  learning_rate_init=0.05)))
models.append(('MLP  2 neuronios, a taxa 0,01 ', MLPClassifier(hidden_layer_sizes=(2, ),  learning_rate_init=0.01)))

models.append(('MLP  5 neuronios, a taxa 0,1 ', MLPClassifier(hidden_layer_sizes=(5, ),  learning_rate_init=0.1)))
models.append(('MLP  5 neuronios, a taxa 0,05 ', MLPClassifier(hidden_layer_sizes=(5, ),  learning_rate_init=0.05)))
models.append(('MLP  5 neuronios, a taxa 0,01 ', MLPClassifier(hidden_layer_sizes=(5, ),  learning_rate_init=0.01)))

models.append(('MLP  10 neuronios, a taxa 0,1 ', MLPClassifier(hidden_layer_sizes=(10, ),  learning_rate_init=0.1)))
models.append(('MLP  10 neuronios, a taxa 0,05 ', MLPClassifier(hidden_layer_sizes=(10, ),  learning_rate_init=0.05)))
models.append(('MLP  10 neuronios, a taxa 0,01 ', MLPClassifier(hidden_layer_sizes=(10, ),  learning_rate_init=0.01)))



# Parametros do scoring http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print("Iniciando cross validation MLP")


for name, model in models:
    print("Model:{0} and n splits = 3".format(name))
    score_acc = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, features, labels_somente_numericos, cv=cvn3, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))

    print("Model:{0} and n splits = 5".format(name))
    score_acc = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, features, labels_somente_numericos, cv=cvn5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))
	

