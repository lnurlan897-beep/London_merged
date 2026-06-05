# Pandas kitabxanasını import edirik
import pandas as pd

# Məlumatı təlim və test hissələrinə bölmək üçün funksiyanı import edirik
from sklearn.model_selection import train_test_split

# XGBoost Regressor modelini import edirik
from xgboost import XGBRegressor

# Modelin performansını ölçmək üçün R² metricini import edirik
from sklearn.metrics import r2_score

import pickle


# CSV faylını oxuyuruq
df = pd.read_csv("london_merged.csv")

# timestamp sütununu datetime formatına çeviririk
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Tarixdən gün məlumatını çıxarırıq
df['day'] = df['timestamp'].dt.day

# Tarixdən saat məlumatını çıxarırıq
df['hour'] = df['timestamp'].dt.hour

# Artıq lazım olmadığı üçün timestamp sütununu silirik
df.drop('timestamp', inplace=True, axis=1)

# Hədəf dəyişəni (cnt) çıxarıb xüsusiyyətləri (features) saxlayırıq
x = df.drop('cnt', axis=1)

# Proqnozlaşdırılacaq sütun
y = df['cnt']

# Dataseti 80% train, 20% test olaraq bölürük
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Model obyektini yaradırıq
xgb = XGBRegressor()

# Modeli təlim məlumatları üzərində öyrədirik
model = xgb.fit(x_train, y_train)

# Test məlumatları üçün proqnozlar alırıq
y_pred = model.predict(x_test)

# R² skorunu hesablayıb ekrana çıxarırıq
print(r2_score(y_test, y_pred))

pickle.dump(model,open('model.pkl','wb'))
