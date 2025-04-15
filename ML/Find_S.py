

import pandas as pd
# Dataset link : https://www.kaggle.com/datasets/fredericobreno/play-tennis

df = pd.read_csv("path")

X = df.drop('play', axis=1)
y = df['play']

hypothesis = ['0'] * len(X.columns)
print("Initial hypothesis : ", hypothesis)



for i in range(len(X)):
    if y.iloc[i] == 'Yes':
        hypothesis = X.iloc[i].tolist()
        break
print("First positive hypothesis : ", hypothesis)

for i in range(len(X)):
    if y.iloc[i] == 'Yes':
        for j in range(len(X.columns)):
            if hypothesis[j] != X.iloc[i, j]:
                hypothesis[j] = '?'

print("Final Hypothesis:", hypothesis)
