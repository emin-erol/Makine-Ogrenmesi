import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veriler = pd.read_csv("bitcoin.csv")

veriler = veriler.sort_values('Date')
# print(veriler.head())

# bagimli degisken y'yi ve bagimsiz degiskenleri olusturuyoruz
x = veriler[['Open', 'High', 'Low', 'Volume BTC', 'Volume USD']]
y = veriler[['Close']]

price = veriler[['Close']]
plt.figure(figsize=(15, 5))
plt.plot(price)
plt.xticks(range(0, veriler.shape[0], 50), veriler['Date'].loc[::50], rotation=90)
plt.title("Bitcoin Fiyatı", fontsize=18, fontweight='bold')
plt.xlabel("Tarih", fontsize=18)
plt.ylabel("Kapanış Fiyatı (USD)", fontsize=18)
plt.show()

scaler = MinMaxScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# bagimsiz degiskenlerin p-value degerlerini gorelim
model1 = sm.OLS(y_train, x_train).fit()
# print(model1.summary())

lr = LinearRegression()
lr.fit(x_train, y_train)
tahminler = lr.predict(x_test)
# print(tahminler)

# gercek ve tahmin verilerinin normallestirilmesi
orijinalVeri = pd.DataFrame(scaler.fit_transform(y_test))
tahminVerileri = pd.DataFrame(scaler.fit_transform(tahminler))

# tahmin verisini, test verisini ve arasindaki hata payini yazdiralim
hata = np.sqrt(np.mean(tahminVerileri - orijinalVeri) ** 2)
# print(tahminVerileri)
# print(orijinalVeri)
# print(hata)  # hatayi 0.003 bulduk


