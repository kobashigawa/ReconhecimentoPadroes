import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv(
    filepath_or_buffer='../adultinho.data',
    header=None, 
    sep=',')

dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

target = dataset[['classe']].values
data=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento','Ganhos de Capital', 'Perdas de Capital','classe']).values

target = [1 if i == ' <=50K' else 0 for i in target]

# Cross validation n splits = 3
cvn3 = StratifiedKFold(n_splits=3, shuffle=True)
cvn3.get_n_splits(data, target)

# Cross validation n splits = 5
cvn5 = StratifiedKFold(n_splits=5, shuffle=True)
cvn5.get_n_splits(data, target)

models = []
models.append(('Naive Bayes Gauss', GaussianNB()))
models.append(('Naive Bayes Multinomial', MultinomialNB()))
models.append(('Naive Bayes Bernoulli', BernoulliNB()))

for name, model in models:
    print("Model:{0} and n splits = 3".format(name))
    score_acc = cross_val_score(model, data, target, cv=cvn3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, data, target, cv=cvn3, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, data, target, cv=cvn3, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))

    print("Model:{0} and n splits = 5".format(name))
    score_acc = cross_val_score(model, data, target, cv=cvn5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_acc.mean(), score_acc.std() * 2))
    score_prec = cross_val_score(model, data, target, cv=cvn5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (score_prec.mean(), score_prec.std() * 2))
    score_recall = cross_val_score(model, data, target, cv=cvn5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (score_recall.mean(), score_recall.std() * 2))
