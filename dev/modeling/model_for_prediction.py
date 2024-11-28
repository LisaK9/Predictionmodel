"""Für die Vorhersage wird GradientBoosting weiter verwendet, da dieses Modell die besten Ergebnisse erzielen konnte
Hier wird evaluiert, wie das Modell auf die zurückgehaltenen noch nicht gesehenen Testdaten vom Mai reagiert.
Des weiteren wird das selbe für ein Full-Training (alle verfügbaren Trainingsdaten) evaluiert.
Da in Zukunft jedoch keine Werte für calls und n_sick vorliegen, wird des weiteren evaluiert, wie
das Modell auf die Eingaben der Schätzungen reagiert, da dies in Zukunft der Live-Prozess sein wird.
Da calls nur sehr schwer vorherzusagen ist, werden die Vorhersagen am Ende mit einem Puffer von 15%
"""
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

data = pd.read_csv('C:\\Users\\Lisa\\PycharmProjects\\Predictionmodel\\dev\\data_preparation\\sickness_modeling.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
test_data = data.loc['2019-05-01':]
print("Testdata: ", test_data)
data = data[:'2019-04-14']

features = ['calls', 'n_duty', 'year', 'n_sick']
target = 'sby_need'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42)
#Trainings-/Validierungsfehler
X_train_best = scaler.transform(X_train_scaled)
y_train_best = y_train

train_sizes, train_scores, test_scores = learning_curve(model, X_train_best, y_train_best, cv=5,
                                                            scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Validation error')
plt.title(f'Learning Curve ')
plt.xlabel('Training examples')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.savefig('learning curve vor Modelltraining')
plt.show()

#Modell trainieren
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred
print("mse: ", mse)

#Vorhersagen evaluieren, die das Modell anhand des Holdout-sets (1-27.05) macht

X_test_data = test_data[features]
y_test_data = test_data[target]
X_test_data_scaled = scaler.transform(X_test_data)

# Vorhersagen auf den neuen Daten machen
y_test_pred = model.predict(X_test_data_scaled)
mse = mean_squared_error(y_test_data, y_test_pred)
mae = mean_absolute_error(y_test_data, y_test_pred)
mse = mean_squared_error(y_test_data, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_data, y_test_pred)
residuals = y_test_data - y_test_pred
print("MSE Holout-Set: ", mse)
print("MAE Holout-Set: ", mae)
print("RMSE Holout-Set: ", rmse)
print("R2 Holout-Set: ", r2)
# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test_pred, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für Holdout Set")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage mit Holdout')
plt.show()






#Evaluieren, wie sich die Schätzungen als Input auf die Vorhersagen des Modells auswirken
data_features = pd.read_csv('features_for_prediction.csv')
X_data_features = data_features[features]
X_data_features_scaled = scaler.transform(X_data_features)

# Vorhersagen auf den neuen Daten machen
y_pred_features = model.predict(X_data_features_scaled)
mse = mean_squared_error(y_test_data, y_pred_features)
mae = mean_absolute_error(y_test_data, y_pred_features)
mse = mean_squared_error(y_test_data, y_pred_features)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_data, y_pred_features)
residuals = y_test_data - y_pred_features
print("MSE Schätzwerte: ", mse)
print("MAE Schätzwerte: ", mae)
print("RMSE Schätzwerte: ", rmse)
print("R2 Schätzwerte: ", r2)

# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_features, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für geschätzte Inputs")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage mit Schätzwerten')
plt.show()

#Ein Full-Training anhand der gesamten verfügbaren Trainingsdaten machen (immer noch bis 14.04.)
#Vorhersagen für Holdout evaluieren

scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X)

model2 = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42)
model2.fit(X_train_scaled_full, y)

y_pred_full = model2.predict(X_test_data_scaled)
mse = mean_squared_error(y_test_data, y_pred_full)
mae = mean_absolute_error(y_test_data, y_pred_full)
mse = mean_squared_error(y_test_data, y_pred_full)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_data, y_pred_full)
residuals = y_test_data - y_pred_full
print("MSE Holdout Fulltraining: ", mse)
print("MAE Holdout FUlltraining: ", mae)
print("RMSE Holdout Fulltraining: ", rmse)
print("R2 Holdout Fulltraining: ", r2)

# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_full, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für Holdout-Set mit Full-Training")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage Full Training mit Hold Out')
plt.show()

#Dasselbe für die geschätzten Features als Input evaluieren

y_pred_features_full = model2.predict(X_data_features_scaled)
print(y_pred_features_full)
mse = mean_squared_error(y_test_data, y_pred_features_full)
mae = mean_absolute_error(y_test_data, y_pred_features_full)
mse = mean_squared_error(y_test_data, y_pred_features_full)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_data, y_pred_features_full)
residuals = y_test_data - y_pred_features_full
print("MSE Schätzwerte Fulltraining: ", mse)
print("MAE Schätzwerte Fulltraining: ", mae)
print("RMSE Schätzwerte Fulltraining: ", rmse)
print("R2 Schätzwerte Fulltraining: ", r2)

# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_features_full, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für geschätzte Inputs nach Full-Training")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage Full Training mit Schätzwerten')
plt.show()


mean_sby_need = np.mean(data['sby_need'])
min_driver = np.ceil(mean_sby_need) #Mindestanzahl an Bereitschaftsfahrern
print("Mean sby_need: ", min_driver)

# Statistische Auswertung der aktuellen Aktivierungsrate
activation_rate_stats = test_data[['activation_rate']].describe()

# Berechnung des Anteils der Werte über 100 % für aktuelle Aktivierungsrate
above_100_actual = (test_data['activation_rate'] > 100).mean() * 100

# Anzahl der Tage und Durchschnittswert für aktuelle Aktivierungsrate über 100%
days_over_100_actual = test_data[test_data['activation_rate'] > 100]

# Anzahl der Tage über 100% für aktuelle Aktivierungsrate
count_over_100_actual = len(days_over_100_actual)

# Durchschnittlicher Wert über 100% für aktuelle Aktivierungsrate
mean_over_100_actual = (days_over_100_actual['activation_rate'] - 100).mean()
# Durchschnittliche aktuelle Aktivierungsrate unter 100 %
below_100_adjusted = test_data[(test_data['activation_rate'] < 100) & (test_data['activation_rate'] > 0)]
mean_below_100 = below_100_adjusted['activation_rate'].mean()
count_below_100 = below_100_adjusted.shape[0]
# Ausgabe bezüglich aktueller Aktivierungsrate
print("Statistische Kennwerte für Aktivierungsraten:\n", activation_rate_stats)
print("\nAnteil der Tage über 100% - Tatsächliche Aktivierungsrate:", above_100_actual, "%")
print("Anzahl der Tage über 100% - Tatsächliche Aktivierungsrate:", count_over_100_actual)
print("Durchschnittswert über 100% - Tatsächliche Aktivierungsrate:", mean_over_100_actual)
print("Anzahl Tage unter 100% - Tatsächliche Aktivierungsrate:", count_below_100)
print("Durchschnittswert unter 100% - Tatsächliche Aktivierungsrate:", mean_below_100)


y_pred_adjusted = np.where(y_pred_features_full < min_driver, min_driver, y_pred_features_full*1.15)  # Mindestens durchschnittliche sby_need als min Fahrer. + 15% Puffer

y_pred_final = np.ceil(y_pred_adjusted)
test_data['y_pred_final'] = y_pred_final
# Aktivierungsrate neu berechnen
test_data['activation_rate_adjusted'] = np.where(
    (test_data['sby_need'] == 0),
    0,
    test_data['sby_need'] / y_pred_final * 100
)

# Statistische Auswertung der neuen Aktivierungsrate
activation_rate_adjusted_stats = test_data[['activation_rate', 'activation_rate_adjusted']].describe()

# Berechnung des Anteils der Werte über 100 % für die neue Aktivierungsrate
above_100_adjusted = (test_data['activation_rate_adjusted'] > 100).mean() * 100
days_over_100_adjusted = test_data[test_data['activation_rate_adjusted'] > 100]
count_over_100_adjusted = len(days_over_100_adjusted)
mean_over_100_adjusted = (days_over_100_adjusted['activation_rate_adjusted'] - 100).mean()
# Durchschnittliche Aktivierungsrate unter 100 % für angepasste Vorhersagen
below_100_adjusted = test_data[(test_data['activation_rate_adjusted'] <= 100) & (test_data['activation_rate_adjusted'] > 0)]
mean_below_100_adjusted = below_100_adjusted['activation_rate_adjusted'].mean()
count_below_100_adjusted = below_100_adjusted.shape[0]

# Ausgabe der statistischen Zusammenfassung und Anteil über 100 % für die neue Aktivierungsrate
print("Statistische Kennwerte für Aktivierungsraten:\n", activation_rate_adjusted_stats)
print("\nAnteil der Tage über 100% - Neue Aktivierungsrate (Adjusted):", above_100_adjusted, "%")
print("Anzahl der Tage über 100% - Neue Aktivierungsrate (Adjusted):", count_over_100_adjusted)
print("Durchschnittswert über 100% - Neue Aktivierungsrate (Adjusted):", mean_over_100_adjusted)
print("Durchschnittswert unter 100% - Neue Aktivierungsrate:", mean_below_100_adjusted)
print("Anzahl Tage unter 100% - Neue Aktivierungsrate:", count_below_100_adjusted)
mean_drivers_when_activation_rate_zero = test_data.loc[test_data['activation_rate_adjusted'] == 0, 'y_pred_final'].mean()
print("Durchschnittliche Anzahl der bereitgehaltenen Fahrer, wenn die Aktivierungsrate 0 ist:", mean_drivers_when_activation_rate_zero)

# Visualisierung der angepassten Vorhersagen
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_final, label='Angepasste Vorhersagen', color='green', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Angepasste Vorhersagen vs. Tatsächliche Werte")
plt.legend()
plt.grid(True)
plt.savefig('Vergleich angepasste Vorhersagen')
plt.show()

# Visualisierung der Aktivierungsraten
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['activation_rate_adjusted'], label='Aktivierungsrate Adjusted', color='green', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['activation_rate'], label='Aktivierungsrate Actual', color='red', marker='x', linestyle='--')
plt.axhline(y=100, color='grey', linestyle='--', label='100% Aktivierung')
plt.xlabel("Datum")
plt.ylabel("Aktivierungsrate")
plt.title("Vergleich der Aktivierungsraten: Adjusted, Actual")
plt.legend()
plt.grid(True)
plt.savefig('Vergleich aller Aktivierungsraten')
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Aktivierungsraten auf der primären y-Achse
ax1.plot(test_data.index, test_data['activation_rate_adjusted'], label='Aktivierungsrate Adjusted', color='green', marker='o', linestyle='-')
ax1.plot(test_data.index, test_data['activation_rate'], label='Aktivierungsrate Actual', color='red', marker='x', linestyle='--')
ax1.axhline(y=100, color='gray', linestyle='--', label='100% Aktivierung')
ax1.set_xlabel("Datum")
ax1.set_ylabel("Aktivierungsrate")
ax1.set_title("Vergleich der Aktivierungsraten und Anzahl der Bereitschaftsfahrer")
ax1.legend(loc='upper left')
ax1.grid(True)

# Sekundärachse für die Balkendiagramme
ax2 = ax1.twinx()
ax2.bar(test_data.index, y_pred_final, color='blue', alpha=0.3, label='Vorhergesagte Bereitschaftsfahrer')
ax2.bar(test_data.index, [90] * len(test_data), color='purple', alpha=0.3, label='Aktuelle Bereitschaftsfahrer')

ax2.set_ylabel("Anzahl der Bereitschaftsahrer")
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

plt.show()



