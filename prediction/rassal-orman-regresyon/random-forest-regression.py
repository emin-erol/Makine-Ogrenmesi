import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

rf_reg = (RandomForestRegressor(n_estimators=10, random_state=0))
rf_reg.fit(X, Y.ravel())

plt.scatter(X, Y)
plt.plot(X, rf_reg.predict(X))
plt.show()

