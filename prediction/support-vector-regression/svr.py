import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli))
plt.show()

