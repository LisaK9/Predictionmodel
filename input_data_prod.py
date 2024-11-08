import pandas as pd
from prophet import Prophet
from datetime import timedelta
import matplotlib.pyplot as plt
import modeling_functions
import holidays

# Daten einlesen und vorbereiten
file_path = 'sickness_modeling.csv'
data = pd.read_csv(file_path)

# Vorbereitung der Regressoren
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['holiday'] = data['holiday'].astype(int)
data['season'] = data['season'].astype(int)
data['day_of_week'] = data['day_of_week'].astype(int)
data['month_day_interaction'] = (data['month'] - 1) * 7 + data['day_of_week']

# Dynamisches Festlegen der Trainingsdaten bis zum 14. Tag des letzten Monats
max_date = data['date'].max()
train_end_date = max_date.replace(day=14)
train_data = data[data['date'] <= train_end_date]

# Dynamisches Festlegen des Vorhersagezeitraums für den gesamten Folgemonat
prediction_start_date = (train_end_date + pd.DateOffset(months=1)).replace(day=1)
prediction_end_date = (prediction_start_date + pd.DateOffset(months=1)) - timedelta(days=1)

# Prophet-Modell für 'calls' vorbereiten
train_prophet = train_data[['date', 'calls', 'holiday', 'month', 'season', 'month_day_interaction']].rename(columns={'date': 'ds', 'calls': 'y'})

# Prophet initialisieren und Regressoren hinzufügen
model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=9, interval_width=0.95, seasonality_mode='multiplicative')
model.add_seasonality(name='monthly', period=30.5, fourier_order=26)
model.add_regressor('holiday')
model.add_regressor('month')
model.add_regressor('season')
model.add_regressor('month_day_interaction')

# Modell trainieren
model.fit(train_prophet)

# Vorbereitung der zukünftigen Daten für die Vorhersage des Folgemonats
de_holidays = holidays.Germany()
future_dates = pd.DataFrame({'ds': pd.date_range(start=prediction_start_date, end=prediction_end_date)})
future_dates['month'] = future_dates['ds'].dt.month
future_dates['holiday'] = future_dates['ds'].apply(lambda x: 1 if x in de_holidays else 0)
future_dates['season'] = future_dates['month'] % 12 // 3 + 1
future_dates['day_of_week'] = future_dates['ds'].dt.weekday
future_dates['month_day_interaction'] = (future_dates['month'] - 1) * 7 + future_dates['day_of_week']

# Prognosen generieren für 'calls'
forecast = model.predict(future_dates)

# Visualisierung der Vorhersage für 'calls'
plt.figure(figsize=(14, 7))
plt.plot(forecast['ds'], forecast['yhat'], label='Vorhergesagte Calls')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.title('Vorhersage der Calls für den Folgemonat')
plt.xlabel('Datum')
plt.ylabel('Calls')
modeling_functions.save_plot("feature_prediction_calls_prod.png")
plt.show()

# Prophet-Modell für 'n_sick' vorbereiten
train_prophet2 = train_data[['date', 'n_sick', 'day_of_week', 'month', 'holiday', 'season']].rename(columns={'date': 'ds', 'n_sick': 'y'})

# Prophet initialisieren und Regressoren für 'n_sick' hinzufügen
model2 = Prophet(changepoint_prior_scale=0.35)
model2.add_seasonality(name='yearly', period=365.25, fourier_order=1)
model2.add_regressor('day_of_week')
model2.add_regressor('holiday')
model2.add_regressor('month')
model2.add_regressor('season')

# Modell für 'n_sick' trainieren
model2.fit(train_prophet2)

# Vorhersagen für 'n_sick' generieren
forecast2 = model2.predict(future_dates)

# Visualisierung der Prognose für 'n_sick'
plt.figure(figsize=(14, 7))
plt.plot(forecast2['ds'], forecast2['yhat'], label='Vorhergesagte n_sick')
plt.fill_between(forecast2['ds'], forecast2['yhat_lower'], forecast2['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.title('Vorhersage der Krankmeldungen für den Folgemonat')
plt.xlabel('Datum')
plt.ylabel('n_sick')
modeling_functions.save_plot("feature_prediction_sick_prod.png")
plt.show()

# Vorhersagen für 'calls' und 'n_sick' in einen DataFrame zusammenführen
combined_forecast_df = pd.DataFrame({
    'date': forecast['ds'],
    'calls': forecast['yhat'],
    'n_sick': forecast2['yhat']
})

# Zusätzliche Features für das Modell erstellen
combined_forecast_df['year'] = combined_forecast_df['date'].dt.year
combined_forecast_df['week'] = combined_forecast_df['date'].dt.isocalendar().week
combined_forecast_df['n_duty'] = 1900  # Fester Wert, manuelle Änderung

# Speichern der Vorhersagen als CSV-Datei
output_path = "input_data_model_prod.csv"
combined_forecast_df.to_csv(output_path, index=False)
print("Vorhersagen gespeichert in:", output_path)
