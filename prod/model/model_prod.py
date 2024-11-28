from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('C:\\Users\\Lisa\\PycharmProjects\\Predictionmodel\\dev\\data_preparation\\sickness_modeling.csv')
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

model = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42) #Modell initialisieren

model.fit(X_train_scaled, y) #Full-Training

# Trainiertes Modell und Scaler speichern
joblib.dump(model, 'C:\\Users\\Lisa\\PycharmProjects\\Predictionmodel\\prod\\model\\trained_gradient_boosting_model.joblib')
print("Das Modell wurde erfolgreich gespeichert.")
joblib.dump(scaler, 'C:\\Users\\Lisa\\PycharmProjects\\Predictionmodel\\prod\\model\\scaler.joblib')
print("Scaler wurde erfolgreich gespeichert.")
