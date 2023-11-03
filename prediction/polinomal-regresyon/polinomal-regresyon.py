import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

# ornek olmasi acisindan once basit dogrusal regresyon olusturup sonucu goruyoruz
linReg = LinearRegression()
linReg.fit(X, Y)

plt.scatter(X, Y)
plt.plot(x, linReg.predict(X))
# plt.show()

# polinomal regresyon modelini olusturuyoruz
polyReg = PolynomialFeatures(degree=2)  # ikinci dereceye kadar cikan bir polinom fonksiyonu olusturduk
xPoly = polyReg.fit_transform(X)  # X dizisini vererek polyReg e gore fit_transform ettik
# print(xPoly)
linReg2 = LinearRegression()
linReg2.fit(xPoly, y)
plt.scatter(X, Y)
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)))
plt.title("Lineer ve Polinomal Regresyon Modelleri")

polyReg2 = PolynomialFeatures(degree=4)  # ikinci dereceye kadar cikan bir polinom fonksiyonu olusturduk
xPoly2 = polyReg2.fit_transform(X)  # X dizisini vererek polyReg e gore fit_trnsform ettik
# print(xPoly)
linReg3 = LinearRegression()
linReg3.fit(xPoly2, y)
plt.scatter(X, Y)
plt.plot(X, linReg3.predict(polyReg2.fit_transform(X)))
plt.title("Lineer ve Polinomal Regresyon Modelleri")
plt.show()

