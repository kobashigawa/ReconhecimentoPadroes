import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
    header=None, 
    sep=',')

df.columns=['Idade', 'Emprego', 'Peso final', 'Educação', 'Anos de Estudo', 'Estado Civil', 'Ocupação', 'Relacionamento', 'Raça', 'Sexo', 'Ganhos de Capital', 'Perdas de Capital', 'Horas de trabalho por semana', 'País de Nascimento', 'classe']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

print(df)


# split data table into data X and class labels y

X = df.loc[:,0:14].values
y = df.loc[:,14].values



traces = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {'50>': 'rgb(31, 119, 180)', 
          '<=50': 'rgb(255, 127, 14)'}

for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col], 
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)

layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

fig = Figure(data=data, layout=layout)
py.iplot(fig)
