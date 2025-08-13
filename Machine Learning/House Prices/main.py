import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv('train.csv')

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
df_encoded.fillna(df_encoded.mean(), inplace=True)

X = df_encoded.drop(['SalePrice', 'Id'], axis=1)
y = df_encoded['SalePrice']
y_log = np.log(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=1000, learning_rate = 0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train_log)

y_pred = model.predict(X_test)
mse_log = mean_squared_error(y_test_log, y_pred)
r2_log = r2_score(y_test_log, y_pred)
scores = cross_val_score(model, X, y_log, cv=5, scoring='r2')

print(f'MSE: {mse_log:.2f}')
print(f'R2: {r2_log:.3f}')
print(f'R2 (cv=5):{scores.mean():.3f} ± {scores.std():.3f}')