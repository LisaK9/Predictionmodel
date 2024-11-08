"""Für die Vorhersage wird GradientBoosting weiter verwendet, da dieses Modell die besten Ergebnisse erzielen konnte
Hier wird evaluiert, wie das Modell auf die zurückgehaltenen noch nicht gesehenen Testdaten vom Mai reagiert.
Des weiteren wird das selbe für ein Full-Training (alle verfügbaren Trainingsdaten) evaluiert.
Da in Zukunft jedoch keine Werte für calls und n_sick vorliegen, wird des weiteren evaluiert, wie
das Modell auf die Eingaben der Schätzungen reagiert, da dies in Zukunft der Live-Prozess sein wird.
Da calls nur sehr schwer vorherzusagen ist, wird anschließend evaluiert, ob die Vorhersagen mit einem zufälligen
Rauschen während des Modelltrainings verbessert werden können. Anschließend wird das finale Modell für
die zukünftige Vorhersage ausgewählt.
"""
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

data = pd.read_csv('sickness_modeling.csv')
data['date'] = pd.to_datetime(data['date'])
data['day'] = data['date'].dt.day
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

#Vorhersagen evaluieren, die das Modell anhand des Test-sets (1-27.05) macht
# Scaler anwenden auf test_data
X_test_data = test_data[features]
y_test_data = test_data[target]
X_test_data_scaled = scaler.transform(X_test_data)

# Vorhersagen auf den neuen Daten machen
y_test_pred = model.predict(X_test_data_scaled)
#y_test_true = test_data[target]
mse = mean_squared_error(y_test_data, y_test_pred)
print("MSE mit testdaten eingabe",mse)
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
print("MSE features",mse)

# Plot der Vorhersagen vs. Tatsächliche Werte (sofern vorhanden)
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
#Vorhersagen für Testdaten evaluieren
scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X)

model2 = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42)
model2.fit(X_train_scaled_full, y)

y_pred_full = model2.predict(X_test_data_scaled)
mse = mean_squared_error(y_test_data, y_pred_full)
print("MSE full test daten eingabe",mse)

# Plot der Vorhersagen vs. Tatsächliche Werte (sofern vorhanden)
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_full, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für Full-Training mit Testdaten")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage Full Training mit Hold Out')
plt.show()

#Dasselbe für die geschätzten Features als Input evaluieren
# Vorhersagen auf den neuen Daten machen
y_pred_features_full = model2.predict(X_data_features_scaled)
mse = mean_squared_error(y_test_data, y_pred_features_full)
print("MSE full features", mse)

# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_features_full, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für geschätzte Werte nach Full-Training")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersage Full Training mit Schätzwerten')
plt.show()


#Evaluieren, ob zufälliges Rauschen für Calls während des Modells für bessere Vorhersagen sorgt,
X_augmented = X.copy()
X_augmented['calls'] = X['calls'] * (1 + np.random.normal(0, 0.01, size=len(X)))
y_augmented = data[target]

X_train_augmented2, X_test_augmented2, y_train_augmented_2, y_test_augmented_2 = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled_augmented2 = scaler.fit_transform(X_train_augmented2)
X_test_scaled_augmented2 = scaler.fit_transform(X_test_augmented2)

model4 = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42)

#Trainings-/Validierungsfehler
X_train_best = scaler.transform(X_train_augmented2)
y_train_best = y_train_augmented_2

train_sizes, train_scores, test_scores = learning_curve(model4, X_train_best, y_train_best, cv=5,
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
plt.savefig('Learning Curve mit Datenrauschen')
plt.show()

model4.fit(X_train_scaled_augmented2, y_train_augmented_2)

y_pred_augmented2 = model4.predict(X_test_scaled_augmented2)
mse = mean_squared_error(y_test_augmented_2, y_pred_augmented2)
print("MSE augmented",mse)


#Full Training mit Datenrauschen
X_train_augmented = X.copy()
X_train_augmented['calls'] = X['calls'] * (1 + np.random.normal(0, 0.01, size=len(X)))
scaler = StandardScaler()
X_train_scaled_augmented = scaler.fit_transform(X_train_augmented)


model3 = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=42)

#Trainings-/Validierungsfehler
X_train_best = scaler.transform(X_train_augmented)
y_train_best = y

train_sizes, train_scores, test_scores = learning_curve(model3, X_train_best, y_train_best, cv=5,
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
plt.savefig('Learning Curve mit Datenrauschen vor Full-Training')
plt.show()

model3.fit(X_train_scaled_augmented, y)

y_pred_augmented = model3.predict(X_test_data_scaled)
mse = mean_squared_error(y_test_data, y_pred_augmented)
print("MSE augmented",mse)

# Plot der Vorhersagen vs. Tatsächliche Werte
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_augmented, label='Vorhergesagte Werte', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['sby_need'], label='Tatsächliche Werte', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("sby_need")
plt.title("Vorhersagen vs. Tatsächliche Werte für geschätzte Werte mit Datenrauschen")
plt.legend()
plt.grid(True)
plt.savefig('Vorhersagen mit Datenrauschen')
plt.show()

#Trainings-/Validierungsfehler
X_train_best = scaler.transform(X_train_augmented)
y_train_best = y

train_sizes, train_scores, test_scores = learning_curve(model3, X_train_best, y_train_best, cv=5,
                                                            scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Validation error')
plt.title(f'Learning Curve after full training')
plt.xlabel('Training examples')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.savefig('Learning Curve mit Datenrauschen - Fulltraining')
plt.show()

# Berechnung der Aktivierungsrate
test_data['activation_rate_new'] = np.where(
    test_data['sby_need'] == 0,  # Bedingung
    0,  # Aktivierungsrate auf 0 setzen, wenn `sby_need` gleich 0 ist
    test_data['sby_need']/y_pred_augmented*100  # Standardberechnung
)
print(test_data.columns)

# Statistische Auswertung der Aktivierungsraten
activation_rate_stats = test_data[['activation_rate', 'activation_rate_new']].describe()

# Berechnung des Anteils der Werte über 100 % für jede Aktivierungsrate
above_100_actual = (test_data['activation_rate'] > 100).mean() * 100
above_100_new = (test_data['activation_rate_new'] > 100).mean() * 100

# Anzahl der Tage und Durchschnittswert für Aktivierungsraten über 100%
days_over_100_actual = test_data[test_data['activation_rate'] > 100]
days_over_100_new = test_data[test_data['activation_rate_new'] > 100]

# Anzahl der Tage über 100% für beide Aktivierungsraten
count_over_100_actual = len(days_over_100_actual)
count_over_100_new = len(days_over_100_new)

# Durchschnittlicher Wert über 100% für beide Aktivierungsraten
mean_over_100_actual = (days_over_100_actual['activation_rate'] - 100).mean()
mean_over_100_new = (days_over_100_new['activation_rate_new'] - 100).mean()

# Ausgabe der statistischen Zusammenfassung und Anteile über 100 %
print("Statistische Kennwerte für Aktivierungsraten:\n", activation_rate_stats)
print("\nAnteil der Tage über 100% - Tatsächliche Aktivierungsrate:", above_100_actual, "%")
print("Anteil der Tage über 100% - Neue Aktivierungsrate:", above_100_new, "%")
# Ergebnisse ausgeben
print("Anzahl der Tage über 100% - Tatsächliche Aktivierungsrate:", count_over_100_actual)
print("Durchschnittswert über 100% - Tatsächliche Aktivierungsrate:", mean_over_100_actual)
print("Anzahl der Tage über 100% - Neue Aktivierungsrate:", count_over_100_new)
print("Durchschnittswert über 100% - Neue Aktivierungsrate:", mean_over_100_new)

# Plot der Aktivierungsraten
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['activation_rate_new'], label='Aktivierungsrate predicted', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['activation_rate'], label='Aktivierungsrate actual', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("Aktivierungsrate")
plt.title("Vorhersagen vs. Tatsächliche Werte für geschätzte Werte mit Datenrauschen")
plt.legend()
plt.grid(True)
plt.savefig('Aktivierungsraten')
plt.show()

# Vorhersagen um 10 % erhöhen und Werte unter 15 durch 15 ersetzen (Puffer)
y_pred_augmented_adjusted = y_pred_augmented * 1.10
y_pred_augmented_adjusted = np.where(y_pred_augmented_adjusted < 15, 15, y_pred_augmented_adjusted)  # Werte unter 15 durch 15 ersetzen

# Aktivierungsrate neu berechnen
test_data['activation_rate_adjusted'] = np.where(
    test_data['sby_need'] == 0,
    0,
    test_data['sby_need'] / y_pred_augmented_adjusted * 100
)

# Statistische Auswertung der neuen Aktivierungsrate
activation_rate_adjusted_stats = test_data[['activation_rate', 'activation_rate_new', 'activation_rate_adjusted']].describe()

# Berechnung des Anteils der Werte über 100 % für die neue Aktivierungsrate
above_100_adjusted = (test_data['activation_rate_adjusted'] > 100).mean() * 100
days_over_100_adjusted = test_data[test_data['activation_rate_adjusted'] > 100]
count_over_100_adjusted = len(days_over_100_adjusted)
mean_over_100_adjusted = (days_over_100_adjusted['activation_rate_adjusted'] - 100).mean()

# Ausgabe der statistischen Zusammenfassung und Anteil über 100 % für die neue Aktivierungsrate
print("Statistische Kennwerte für Aktivierungsraten:\n", activation_rate_adjusted_stats)
print("\nAnteil der Tage über 100% - Neue Aktivierungsrate (Adjusted):", above_100_adjusted, "%")
print("Anzahl der Tage über 100% - Neue Aktivierungsrate (Adjusted):", count_over_100_adjusted)
print("Durchschnittswert über 100% - Neue Aktivierungsrate (Adjusted):", mean_over_100_adjusted)

# Visualisierung der angepassten Vorhersagen
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_pred_augmented_adjusted, label='Angepasste Vorhersagen', color='green', marker='o', linestyle='-')
plt.plot(test_data.index, y_pred_augmented, label='Vorhergesagte Werte (Original)', color='blue', marker='o', linestyle='-')
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
plt.plot(test_data.index, test_data['activation_rate_new'], label='Aktivierungsrate Predicted', color='blue', marker='o', linestyle='-')
plt.plot(test_data.index, test_data['activation_rate'], label='Aktivierungsrate Actual', color='red', marker='x', linestyle='--')
plt.xlabel("Datum")
plt.ylabel("Aktivierungsrate")
plt.title("Vergleich der Aktivierungsraten: Adjusted, Predicted, Actual")
plt.legend()
plt.grid(True)
plt.savefig('Vergleich aller Aktivierungsraten')
plt.show()



