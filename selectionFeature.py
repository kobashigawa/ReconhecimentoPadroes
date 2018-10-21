import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(
    filepath_or_buffer='adult.data',
    header=None, 
    sep=',')
	
dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

dataset.dropna(how="all", inplace=True) # drops the empty line at file-end

somente_numericos=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

somente_numericos = somente_numericos.sort_values(by=['classe'])

features_full=somente_numericos[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]
features=np.array(features_full.values, dtype=np.float64)

labels=somente_numericos[['classe']].values

labels_somente_numericos = [1 if i == ' <=50K' else 0 for i in labels]

#print (features)
#print (labels)

#clf = make_pipeline(ReliefF(n_features_to_select=4, n_neighbors=4), RandomForestClassifier(n_estimators=5))

#score = cross_val_score(clf, features, labels, cv=4)

#print(score)

#print(np.mean(score))


X_train, X_test, y_train, y_test = train_test_split(features, labels_somente_numericos)



print(X_train)
print(X_test)
print(y_train)
print(y_test)



fs = ReliefF()
fs.fit(X_train, y_train)

for feature_name, feature_score in zip(features_full.columns,
                                       fs.feature_importances_):
    print(feature_name, '\t', feature_score)