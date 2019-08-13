import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("train.csv")

if (df.isnull().sum().sum() == 0):
    print('Dataframe does not have any null values')
X = df.drop('price_range', axis=1)
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pre = knn.predict(X_test)

y_array = np.round(y_pre, decimals=0)

print("Do chinh xac: ", knn.score(X_test, y_test))

print('Gia tri du doan cua Y tu tep X_test:', y_array.astype(int)[1:20])

print("Gia tri cua tap y_test             :", y_test[1:20].values)
