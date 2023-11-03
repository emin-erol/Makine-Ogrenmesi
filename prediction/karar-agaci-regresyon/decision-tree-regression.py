import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)

plt.scatter(X, Y)
plt.plot(X, dt_reg.predict(X))
plt.show()

