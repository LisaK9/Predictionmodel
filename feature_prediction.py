"""Da die Features calls und n_sick für die Zukunft unbekannt sind, werden diese hier für den Testzeitraum
1.05-27.05.2019 geschätzt und am Ende eine CSV-Datei erstellt, die als Eingabe für das trainierte Modell dienen soll,
um Vorhersagen für diesen Zeitraum zu machen.
Somit kann bereits im Offline-Betrieb evaluiert werden, wie das Modell auf die teils ungenauen Schätzungen
von calls reagiert und trotzdem valide Vorhersagen liefern kann"""

import pandas as pd
from prophet import Prophet
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import modeling_functions

# Daten einlesen
file_path = 'sickness_modeling.csv'
data = pd.read_csv(file_path)

# Vorbereitung der Regressoren
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['day'] = data['date'].dt.day
data['month'] = data['month'].astype(int)
data['holiday'] = data['holiday'].astype(int)
data['season'] = data['season'].astype(int)
data['day'] = data['day'].astype(int)

# Lags hinzufügen
for lag in range(1, 31):  # Lags für die letzten 7 Tage
    data[f'lag_{lag}'] = data['calls'].shift(lag)

for lag in range(1, 31):  # Lags für die letzten 7 Tage
    data[f'lag_sby_need_{lag}'] = data['sby_need'].shift(lag)
print(data)

for lag in range(1, 31):  # Lags für die letzten 7 Tage
    data[f'lag_sick_{lag}'] = data['n_sick'].shift(lag)
print(data)

# Fehlende Werte aufgrund der Lags entfernen
data.dropna(inplace=True)

# Testzeitraum festlegen:

test_start_date = datetime(2019, 5, 1)
test_end_date = datetime(2019, 5, 27)

# Aufteilen in Trainings- und Testdaten
train_data = data[(data['date'] < test_start_date) | (data['date'] > test_end_date)]
test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]


"""Hier wurden mehrere Kombinationen aus zusätzlichen Regressoren, Lag-Features, Säsonalitäten etc. 
getestet. Die besten Ergebnisse konnte die hier vorhandene Kombination erzielen"""

# Zeitliche Variablen und Lags als Zusatzregressoren hinzufügen
train_prophet = train_data[['date', 'calls', 'month','holiday', 'season', 'lag_30','lag_sby_need_30', 'lag_sick_30', 'day']].rename(columns={'date': 'ds', 'calls': 'y'})

# Prophet initialisieren
model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=9, interval_width=0.95, seasonality_mode='multiplicative')
model.add_seasonality(name='monthly', period=30.5, fourier_order=20)
model.add_regressor('holiday')
model.add_regressor('month')
model.add_regressor('season')
model.add_regressor('lag_30')
model.add_regressor('lag_sby_need_30')
model.add_regressor('lag_sick_30')
model.add_regressor('day')

# Modell trainieren
model.fit(train_prophet)


# Vorbereitung der Daten für die Prognose
future_dates = test_data[['date', 'holiday', 'month','season', 'lag_30','lag_sby_need_30','lag_sick_30', 'day']].rename(columns={'date': 'ds'})

# Prognosen generieren
forecast = model.predict(future_dates)

# Berechnung von MSE, MAE und RMSE
mse = mean_squared_error(test_data['calls'], forecast['yhat'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data['calls'], forecast['yhat'])
print("Vorhersagen: ", forecast)
# Ergebnisse anzeigen
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# Ergebnisse visualisieren
plt.figure(figsize=(14, 7))
plt.plot(test_data['date'], test_data['calls'], label='Tatsächliche Calls')
plt.plot(forecast['ds'], forecast['yhat'], label='Vorhergesagte Calls')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.title('Vorhersage der Calls im Vergleich zu den tatsächlichen Werten')
plt.xlabel('Datum')
plt.ylabel('Calls')
modeling_functions.save_plot("feature_prediction_calls.png")
plt.show()



"""Schätzung des Feature n_sick. Hier wurden ebenfalls mehrere Kombinationen getestet"""

train_prophet2 = train_data[['date', 'n_sick', 'day_of_week', 'month','holiday', 'season', 'lag_30','lag_sby_need_30', 'lag_sick_30']].rename(columns={'date': 'ds', 'n_sick': 'y'})

# Prophet initialisieren
model2 = Prophet(changepoint_prior_scale=0.35)
model2.add_seasonality(name='yearly', period=365.25, fourier_order=1)
model2.add_regressor('day_of_week')
model2.add_regressor('holiday')
model2.add_regressor('month')
model2.add_regressor('season')

# Modell trainieren
model2.fit(train_prophet2)

# Vorbereitung der Daten für die Prognose von n_sick
future_dates2 = test_data[['date', 'day_of_week', 'holiday', 'month','season', 'lag_30','lag_sby_need_30','lag_sick_30']].rename(columns={'date': 'ds'})
print('futuredates: ',future_dates2)
# Prognosen generieren
forecast2 = model2.predict(future_dates2)
print('vorhersagen: ', forecast2)
# Berechnung von MSE, MAE und RMSE
mse2 = mean_squared_error(test_data['n_sick'], forecast2['yhat'])
rmse2 = np.sqrt(mse)
mae2 = mean_absolute_error(test_data['n_sick'], forecast2['yhat'])


# Ergebnisse anzeigen
print(f'MSE: {mse2:.2f}')
print(f'RMSE: {rmse2:.2f}')
print(f'MAE: {mae2:.2f}')

# Ergebnisse visualisieren
plt.figure(figsize=(14, 7))
plt.plot(test_data['date'], test_data['n_sick'], label='Tatsächliche n_sick')
plt.plot(forecast2['ds'], forecast2['yhat'], label='Vorhergesagte n_sick')
plt.fill_between(forecast2['ds'], forecast2['yhat_lower'], forecast2['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.title('Vorhersage der Krankmeldungen im Vergleich zu den tatsächlichen Werten')
plt.xlabel('Datum')
plt.ylabel('n_sick')
modeling_functions.save_plot("feature_prediction_sick.png")
plt.show()

# Vorhersagen für calls und n_sick in einen DataFrame zusammenführen
combined_forecast_df = pd.DataFrame({
    'date': forecast['ds'],
    'calls': forecast['yhat'],
    'n_sick': forecast2['yhat']
})

#zusätzlich benötigte Features für das Modell erstellen
combined_forecast_df['year']= combined_forecast_df['date'].dt.year
combined_forecast_df['week']= combined_forecast_df['date'].dt.week
combined_forecast_df['n_duty']= 1900 #wird als fixer Wert betrachtet, da dieser sich nur sehr selten ändert und dann manuell angepasst werden kann

combined_forecasts_df = combined_forecast_df[['date','calls','year','week', 'n_duty','n_sick']]

combined_forecasts_df.set_index('date', inplace=True)
print(combined_forecasts_df)
# Speichern der Vorhersagen als CSV-Datei
output_path = "features_for_prediction.csv"
combined_forecasts_df.to_csv(output_path)