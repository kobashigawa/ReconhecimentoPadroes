import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

dataset = pd.read_csv(
    filepath_or_buffer='adult.data',
    header=None, 
    sep=',')

dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

dataset.dropna(how="all", inplace=True) # drops the empty line at file-end

somente_numericos=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

somente_numericos = somente_numericos.sort_values(by=['classe'])

#Removendo ganhos e perdas de capital através do Relief 8%
features_full=somente_numericos[['Idade', 'Anos de Estudo', 'Horas de trabalho por semana', 'Ganhos de Capital']]

features=np.array(features_full.values, dtype=np.float64)

labels=somente_numericos[['classe']].values

# Necessario usar 0 e 1 para recall e precision
# https://stackoverflow.com/questions/39187875/scikit-learn-script-giving-vastly-different-results-than-the-tutorial-and-gives
# Ha 3 formas descritas no link
labels_somente_numericos = [1 if i == ' <=50K' else 0 for i in labels]

print("Finalizando setup")

# Cross validation n splits = 3
cvn3 = StratifiedKFold(n_splits=3, shuffle=True)
cvn3.get_n_splits(features, labels_somente_numericos)

# Cross validation n splits = 5
cvn5 = StratifiedKFold(n_splits=5, shuffle=True)
cvn5.get_n_splits(features, labels_somente_numericos)

print("Montando o cross validation")

##############################################################################
# Kernel http://scikit-learn.org/stable/modules/svm.html#svm-kernels
# Para nossos testes usamos o SVC com linear kernel, pois nao necessitamos da flexibilidade toda do linearsvc
# https://datascience.stackexchange.com/questions/25046/what-is-the-difference-between-linear-svm-and-svm-with-linear-kernel
##############################################################################
# SVM http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC
# DEFAULT RBF
# GAMMA default 1 / n_features - RBF POLY SIGMOID
# COEF0 default 0 - POLY SIGMOID
# DEGREE default 3 - POLY
##############################################################################
models = []
# KERNEL LINEAR
models.append(('LinearSVC_C0.1', SVC(kernel='linear', C=0.1, gamma=0.1)))
models.append(('LinearSVC_C0.5', SVC(kernel='linear', C=0.5, gamma=0.1)))
models.append(('LinearSVC_C10', SVC(kernel='linear', C=10, gamma=0.1)))
# KERNEL RBF - Gaussian
#models.append(('RbfSVC_C0.1', SVC(kernel='rbf', C=0.1, gamma=0.1)))
#models.append(('RbfSVC_C0.5', SVC(kernel='rbf', C=0.5, gamma=0.1)))
#models.append(('RbfSVC_C10', SVC(kernel='rbf', C=10, gamma=0.1)))
# KERNEL SIGMOID 
#models.append(('SigmoidSVC_C0.1', SVC(kernel='sigmoid', C=0.1, gamma=0.1)))
#models.append(('SigmoidSV_C0.5', SVC(kernel='sigmoid', C=0.5, gamma=0.1)))
#models.append(('SigmoidSV_C10', SVC(kernel='sigmoid', C=10, gamma=0.1)))
# KERNEL POLYNOMIAL
#models.append(('PolynomialSVC_C0.1', SVC(kernel='poly', C=0.1)))
#models.append(('PolynomialSV_C0.5', SVC(kernel='poly', C=0.5)))
#models.append(('PolynomialSV_C10', SVC(kernel='poly', C=10)))

# Parametros do scoring http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print("Realizando o cross validation do SVM")
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

