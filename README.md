# Housing-Price-Dataset-Factors-Affecting-Home
Ev fiyatı tahmini projesi (Machine Learning)

Bu proje, çeşitli ev özelliklerini kullanarak bir konutun satış fiyatını makine öğrenmesi teknikleriyle tahmin etmeyi amaçlamaktadır. Model eğitimi için “Housing Price Dataset” veri seti kullanılmıştır.

Projede veri analizi, veri ön işleme, özellik mühendisliği, model eğitimi, model karşılaştırmaları ve değerlendirme metrikleri uygulanmıştır.

Kodlar Python ile yazılmıştır ve pandas, numpy, scikit-learn, matplotlib, seaborn gibi kütüphaneler kullanılmıştır.

PROJENİN AŞAMALARI

Aşağıda proje içerisinde yapılan adımlar detaylı şekilde açıklanmıştır.

## 1) Kütüphanelerin Yüklenmesi

Aşağıdaki kütüphaneler veri işleme, model eğitimi ve görselleştirme için kullanıldı.

# Kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

## Veri Setinin İndirilmesi

Kaggle veri setini kagglehub üzerinden indirip çalışma ortamına aldık.

!pip install kagglehub

import kagglehub

# Download latest version
path = kagglehub.dataset_download("aungpyaeap/housing")

print("Path to dataset files:", path)

## 3) Veri Setinin Yüklenmesi ve İlk İnceleme
# Dataset yükle
df = pd.read_csv("/kaggle/input/housing/Housing.csv")

# İlk 5 satır
df.head()

# Veri tipi ve sütun bilgileri
df.info()

# Eksik değer kontrolü
missing_values = df.isnull().sum()
print('Missing values:')
print(missing_values)


Bu aşamada veri setinin eksiksiz olduğu görülmüştür.

## 4) Kategorik Değişkenlerin Dönüştürülmesi

Veri setindeki birçok sütun “yes/no” şeklindedir. Bunlar modele girebilmesi için 0/1 formatına dönüştürüldü.

binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})


Ayrıca furnishingstatus one-hot encoding ile dönüştürüldü:

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

## 5) Özellik Mühendisliği

Ev kalitesini yükseltmek amacıyla iki yeni özellik oluşturuldu:

df['amenities'] = df['parking'] + df['airconditioning'] + df['hotwaterheating']
df['location_quality'] = np.where(df['prefarea'] == 1, 'good', 'normal')
df = pd.get_dummies(df, columns=['location_quality'], drop_first=True)

## 6) Veri Setinin X ve y Olarak Ayrılması
X = df.drop(['price'], axis=1)
y = df['price']

## 7) Veri Setinin Eğitim ve Test Olarak Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 8) Sayısal Değişkenlerin Ölçeklendirilmesi

Makine öğrenmesi modellerinin daha iyi çalışması için StandardScaler uygulandı.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 9) Kullanılan Modeller

Projede üç farklı algoritma çalıştırıldı:

1-Random Forest
2-Gradient Boosting
3-Ridge Regression
models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Ridge Regression": Ridge()
}


Her model için eğitim ve değerlendirme yapıldı:

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

## 10) Model Sonuçları

Modellerin başarı seviyeleri karşılaştırıldı:

En düşük RMSE → Gradient Boosting

En yüksek R² → Random Forest

En stabil model → Ridge Regression

## 11) Özellik Önemlerinin Görselleştirilmesi (Random Forest)
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)

importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 8))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.show()


Bu grafik ev fiyatını etkileyen en önemli değişkenleri net şekilde göstermektedir.
## SONUÇ

Bu projede konutların satış fiyatlarını tahmin etmek için makine öğrenmesi algoritmaları uygulanmıştır.
Veri ön işleme, özellik mühendisliği, model eğitimi ve karşılaştırması yapılmış ve en başarılı model belirlenmiştir.

Proje hem veri analizi hem de modelleme açısından temel makine öğrenmesi uygulamalarını içermektedir.
