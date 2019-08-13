import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("train.csv")

if (df.isnull().sum().sum() == 0):
    print('Data frame does not have any null values')

X = df.drop(['price_range'], axis=1)
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)
print('Intercept (B0): \n', lm.intercept_)
print('Coefficients (B1-Bn): \n', lm.coef_)
print('Accuracy: \n', lm.score(X_train, y_train))

y_pre = lm.predict(X_test)

y_array = np.round(y_pre, decimals=0)
print('Gia tri du doan cua Y tu tep X_test:', y_array.astype(int)[1:20])

print("Gia tri cua tap y_test             :", y_test[1:20].values)
