import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

veriler = pd.read_csv("odev_tenis.csv")
# print(veriler)

# verilerimiz kategorik kolonlara sahip oldugu icin oncelikle o kolonlari one hot encoding ederek sayisal hale getirelim
# sayisal hale getirecegimiz kolonlari alalim
havaDurumu = veriler.iloc[:, 0:1].values
ruzgarDurumu = veriler.iloc[:, 3:4].values
oyunDurumu = veriler.iloc[:, -1:].values
# print(havaDurumu)
# print(ruzgarDurumu)
# print(oyunDurumu)

# hava durumu kolonunu sayisal hale getirelim
le = preprocessing.LabelEncoder()
havaDurumu[:, 0] = le.fit_transform(veriler.iloc[:, 0])
# print(havaDurumu)
# sayisal hale gelen kolonu ohe islemi ile kolonlara ayiralim
ohe = preprocessing.OneHotEncoder()
havaDurumu = ohe.fit_transform(havaDurumu).toarray()
# print(havaDurumu)

# ruzgar durumu kolonunu sayisal hale getirelim
le = preprocessing.LabelEncoder()
ruzgarDurumu[:, 0] = le.fit_transform(veriler.iloc[:, 3])
# print(ruzgarDurumu)
# sayisal hale gelen kolonu ohe islemi ile kolonlara ayiralim
ohe = preprocessing.OneHotEncoder()
ruzgarDurumu = ohe.fit_transform(ruzgarDurumu).toarray()
# print(ruzgarDurumu)

# oyun durumu kolonunu sayisal hale getirelim
le = preprocessing.LabelEncoder()
oyunDurumu[:, 0] = le.fit_transform(veriler.iloc[:, 4])
# print(oyunDurumu)
# sayisal hale gelen kolonu ohe islemi ile kolonlara ayiralim
ohe = preprocessing.OneHotEncoder()
oyunDurumu = ohe.fit_transform(oyunDurumu).toarray()
# print(oyunDurumu)

# one hot encoding isleminden sonra  dosyamiza bu olusan numerik kolonlari ekleyelim

sHavaDurumu = pd.DataFrame(data=havaDurumu, index=range(14), columns=['overcast', 'rainy', 'sunny'])
# print(sHavaDurumu)
sRuzgarDurumu = pd.DataFrame(data=ruzgarDurumu[:, 1:2], index=range(14), columns=['ruzgar'])
# print(sRuzgarDurumu)
sOyunDurumu = pd.DataFrame(data=oyunDurumu[:, 1:2], index=range(14), columns=['oyun'])
# print(sOyunDurumu)

# tahmin ettirecegimiz nem kolonunu ayiriyoruz
nemDurumu = veriler.iloc[:, 2:3]
# print(nemDurumu)

# datasetimizdeki kategorik verileri kaldirip yerine sayisal hale gecirdigimiz kolonlari ekliyoruz
veriler.drop('outlook', axis=1, inplace=True)
veriler.drop("windy", axis=1, inplace=True)
veriler.drop("play", axis=1, inplace=True)
veriler = pd.concat([sHavaDurumu, veriler], axis=1)
veriler = pd.concat([veriler, sRuzgarDurumu], axis=1)
veriler = pd.concat([veriler, sOyunDurumu], axis=1)

# datasetinin yeni halinden train icin kullanilacak kolonlari verilerTrain degiskenine aliyoruz
verilerTrain = veriler.drop('humidity', axis=1)
print(verilerTrain)

# duzenledigimiz datasetini egitim ve test kumelerine ayiriyoruz
x_train, x_test, y_train, y_test = train_test_split(verilerTrain, nemDurumu, test_size=0.33, random_state=0)

# dogrusal regresyon modeli olusturup icine egitim kumelerini veriyoruz
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# egitilen regresyon modeline tahmin etmesi icin x_test verilerini veriyoruz
predict = regressor.predict(x_test)

# print("Veri Setindeki y_test:")
# print(y_test)
# print("\nTahmin Dizisi:")
# print(predict)

# basariyi artirmak adina hangi kolonun sistemi nasil etkiledigine bakip kotu etkileyenleri elimine ediyoruz
X = np.append(arr=np.ones((14, 1)).astype(int), values=verilerTrain, axis=1)  # x_train verisine 1 lerden olusan kolon ekliyoruz
XListe = verilerTrain.iloc[:, [0, 1, 2, 3, 4, 5]].values
XListe = np.array(XListe, dtype=float)
model = sm.OLS(nemDurumu, XListe).fit()
print(model.summary())

# 5. kolonun p-valuesi yuksek oldugundan cikarma karari aldik ve tekrardan olctuk
XListe = verilerTrain.iloc[:, [0, 1, 2, 3, 5]].values
XListe = np.array(XListe, dtype=float)
model = sm.OLS(nemDurumu, XListe).fit()
print(model.summary())


