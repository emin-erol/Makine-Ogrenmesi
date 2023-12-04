import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logReg = LogisticRegression(random_state=0)
logReg.fit(X_train, y_train.ravel())
y_pred = logReg.predict(X_test)

print(y_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)

print(cm)



