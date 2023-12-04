import pandas as pd
from sklearn. model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

veriler = pd.read_csv("../veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logReg = LogisticRegression()
logReg.fit(X_train, y_train.ravel())
logPredict = logReg.predict(x_test)

cm1 = confusion_matrix(y_test, logPredict)

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train.ravel())
knnPredict = knn.predict(x_test)

cm2 = confusion_matrix(y_test, knnPredict)

print(cm1)
print(cm2)



