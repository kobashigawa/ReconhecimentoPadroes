import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(
    filepath_or_buffer='adult.data',
    header=None, 
    sep=',')

tamanho_considerado = 12500

dataset.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']

dataset.dropna(how="all", inplace=True) # drops the empty line at file-end

somente_numericos=dataset.drop(columns=['Emprego', 'Educação', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'País de Nascimento'])

features_full=somente_numericos[['Idade', 'Peso final', 'Anos de Estudo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana']]

features=np.array(features_full.values, dtype=np.float64)

#reduzindo tamanho do dataset para nao ultrapassar o limite
features = np.delete(features, np.s_[tamanho_considerado:], axis=0)

labels=somente_numericos[['classe']].values
labels_somente_numericos = [1 if i == ' <=50K' else 0 for i in labels]
labels_somente_numericos = np.array(labels_somente_numericos)
labels_somente_numericos = np.delete(labels_somente_numericos, np.s_[tamanho_considerado:], axis=0)

X_train, X_test, y_train, y_test = train_test_split(features, labels_somente_numericos)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

fs = ReliefF()
fs.fit(X_train, y_train)

for feature_name, feature_score in zip(features_full.columns,
                                       fs.feature_importances_):
    print(feature_name, '\t', feature_score)