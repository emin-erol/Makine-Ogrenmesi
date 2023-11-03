import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


veriler = pd.read_csv('satislar.csv')

aylar = veriler[["Aylar"]]
satislar = veriler[["Satislar"]]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

plt.plot(x_train, y_train)
