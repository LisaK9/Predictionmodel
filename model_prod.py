from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

data = pd.read_csv('sickness_modeling.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
max_date = data.index.max() # Max-Datum bestimmen
max_date_limit = max_date.replace(day=14)  # Tag auf den 14. setzen. Dies dient ausschließlich zur Sicherheit, normalerweise sollte das Max-Datum ausreichen
filtered_data = data[data.index <= max_date_limit] # Daten bis zum 14. des Max-Datums fürs Training
print(filtered_data)

features = ['calls', 'year', 'n_duty', 'n_sick']
target = 'sby_need'

X = filtered_data[features]
y = filtered_data[target]

X_train_augmented = X.copy()
X_train_augmented['calls'] = X['calls'] * (1 + np.random.normal(0, 0.01, size=len(X))) #Datenrauschen für calls einfügen

scaler = StandardScaler()
X_train_scaled_augmented = scaler.fit_transform(X_train_augmented) # Daten skalieren

model = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42) #Modell initialisieren

model.fit(X_train_scaled_augmented, y) #Full-Training

# Trainiertes Modell und Scaler speichern
joblib.dump(model, 'trained_gradient_boosting_model.joblib')
print("Das Modell wurde erfolgreich gespeichert.")
joblib.dump(scaler, 'scaler.joblib')
print("Scaler wurde erfolgreich gespeichert.")