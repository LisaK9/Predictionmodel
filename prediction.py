"""Mit dem trainierten Modell Vorhersagen für den zukünftigen Monat machen. Verwendet werden dafür
als Eingaben die geschätzten Werte für calls und n_sick für den zukünftigen Monat"""
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Eingabedaten für das Modell laden
future_data = pd.read_csv('input_data_model_prod.csv')

# Sicherstellen, dass das Datum als Datetime-Objekt vorliegt
future_data['date'] = pd.to_datetime(future_data['date'])
features = ['calls', 'year', 'n_duty', 'n_sick']
X = future_data[features]
# Verwendeter Scaler laden und benötigte Eingaben für das Modell skalieren
scaler = joblib.load('scaler.joblib')
future_data_scaled = scaler.transform(X)

# Trainiertes Modell laden und Vorhersagen mit den neuen Inputs generieren
model = joblib.load('trained_gradient_boosting_model.joblib')
predictions = model.predict(future_data_scaled)

# Ergebnisse anzeigen
future_data['predicted_sby_need'] = predictions

# Anpassungen: 10% auf die Vorhersagen aufrechnen und Werte unter 15 auf 15 setzen (Dient als Puffer)
predictions_adjusted = predictions * 1.10
predictions_adjusted = np.where(predictions_adjusted < 15, 15, predictions_adjusted)
predictions_adjusted = np.ceil(predictions_adjusted)
# Ergebnisse speichern
future_data['predicted_sby_need_adjusted'] = predictions_adjusted

# Visualisierung der Vorhersagen und angepassten Vorhersagen
plt.figure(figsize=(14, 7))
plt.plot(future_data['date'], future_data['predicted_sby_need'], label='Ursprüngliche Vorhersagen', color='blue', linestyle='--', marker='o')
plt.plot(future_data['date'], future_data['predicted_sby_need_adjusted'], label='Korrigierte Vorhersagen (10% Erhöhung, min. 15)', color='green', linestyle='-', marker='x')
plt.xlabel('Datum')
plt.ylabel('sby_need')
plt.title('Vergleich der ursprünglichen und korrigierten Vorhersagen')
plt.legend()
plt.grid(True)
plt.savefig('Predicted sby_need')
plt.show()

output_path = "predictions.csv"
future_data.to_csv(output_path, index=False)