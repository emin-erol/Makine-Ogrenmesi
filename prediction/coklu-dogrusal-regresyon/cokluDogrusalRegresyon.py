import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

veriler = pd.read_csv("veriler.csv")

# ulke ve cinsiyet kolonundaki kategorik verileri sayisal hale getirelim
ulke = veriler.iloc[:, 0:1].values

# kolondaki verileri okumak icin gerekli fonksiyon
le = preprocessing.LabelEncoder()
# kolondaki degerleri alip sayisal hale getirdik
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
# print(ulke)

# sayisal hale gelen kolonu 0 ve 1 lerden olusabilmesi icin gerekli islem
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
# print(ulke)

cinsiyet = veriler.iloc[:, -1:].values

# kolondaki verileri okumak icin gerekli fonksiyon
le = preprocessing.LabelEncoder()
# kolondaki degerleri alip sayisal hale getirdik
cinsiyet[:, -1] = le.fit_transform(veriler.iloc[:, -1])
# print(cinsiyet)

# sayisal hale gelen kolonu 0 ve 1 lerden olusabilmesi icin gerekli islem
ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
# print(cinsiyet)

# one hot encoding isleminden sonra veriler.csv dosyasina bu olusan numerik kolonlari ekleyelim

kiloVeYas = veriler.iloc[:, 2:4]
boylar = veriler.iloc[:, 1:2]
print(boylar)

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
# print(sonuc)

sonuc2 = pd.DataFrame(data=kiloVeYas, index=range(22), columns=['kilo', 'yas'])
# print(sonuc2)

sonuc3 = pd.DataFrame(data=cinsiyet[:, :1], index=range(22), columns=['cinsiyet'])
# print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
s1 = pd.concat([s, sonuc3], axis=1)
# print(s1)

x_train, x_test, y_train, y_test = train_test_split(s1, boylar, test_size=0.33, random_state=0)

sc = StandardScaler()  # verileri olceklendirme islemi

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# dogrusal regresyon modeli olusturup icine egitim kumelerini veriyoruz
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# egitilen regresyon modeline tahmin etmesi icin x_test verilerini veriyoruz
predict = regressor.predict(x_test)

# basariyi artirmak adina hangi kolonun sistemi nasil etkiledigine bakip kotu etkileyenleri elimine ediyoruz
X = np.append(arr=np.ones((22, 1)).astype(int), values=s1, axis=1)  # x_train verisine 1 lerden olusan kolon ekliyoruz
XListe = s1.iloc[:, [0, 1, 2, 3, 4, 5]].values
XListe = np.array(XListe, dtype=float)
model = sm.OLS(boylar, XListe).fit()
# print(model.summary())  # 4. indisteki kolonun p-value degeri 0.717 ciktigi icin bu kolonu kaldiriyoruz

XListe = s1.iloc[:, [0, 1, 2, 3, 5]].values
XListe = np.array(XListe, dtype=float)
model = sm.OLS(boylar, XListe).fit()
# print(model.summary())





