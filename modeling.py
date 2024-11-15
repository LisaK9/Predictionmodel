import modeling_functions
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

#Daten laden
data = modeling_functions.load_and_preprocess_data('sickness_modeling.csv')
print(data)

#Features und Zielvariable definieren
X, y, features = modeling_functions.define_features_and_target(data)
print("Features: ", features)
#Train Test Split
X_train, X_test, y_train, y_test = modeling_functions.split_data(X, y)

#Speichern der Testwerte zum späteren Vergleich
result_df=y_test.copy()
result_df=pd.DataFrame(result_df)
result_df=result_df.reset_index()

#baseline Modell
results={}
predictions_dict = {}
residuals_dict={}
baseline = LinearRegression()
X_train_scaled_base, X_test_scaled_base, scaler_base = modeling_functions.scale_features(X_train, X_test, features)
mae, mse, rmse, r2, residuals, y_pred = modeling_functions.train_model_without_tuning(baseline, X_train_scaled_base,y_train,X_test_scaled_base,y_test)
print("Baseline MSE: ", mse)
print("Baseline MAE: ", mae)
print("Baseline RMSE: ", rmse)
print("Baseline r2: ", r2)

results['baseline'] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
residuals_dict['baseline'] = residuals
predictions_dict['baseline'] = y_pred

# Modelle und Hyperparameter definieren
models, param_grid = modeling_functions.define_models_and_params()

#Feature Selectors
feature_selectors = modeling_functions.define_feature_selectors()

#Feature-Selektion, Modell-Training, Hyperparameter-Tuning
best_estimators = {}
best_scalers = {}
best_selectors = {}
best_params = {}
model_best_combinations = {}
for model_name, model in models.items():
        print(f"\nTraining for model: {model_name}")
        score = float('inf')
        best_score = float('inf')
        best_r2 = float('inf')
        best_mae = float('inf')
        best_rmse = float('inf')
        best_fs_name = ""
        best_n_features = 0
        best_model = None
        best_scaler = None
        best_selected_features = []
        best_predictions = None
        best_residuals = None

        for fs_name, selector in feature_selectors.items(): #Feature Selektion
            print(f"  Using feature selection method: {fs_name}")

            for n_features in range(1, len(features) + 1): #Jede Anzahl an Features durchlaufen
                print(f"    Selecting {n_features} features with method {fs_name}...")
                selected_feature_indices = modeling_functions.feature_selection(fs_name,selector,X_train,
                                                                         y_train,n_features)
                print("Selected Future Indices: ", selected_feature_indices)
                selected_features = [features[i] for i in selected_feature_indices]
                print(f"      Selected features ({len(selected_features)}): {selected_features}")

                X_train_scaled, X_test_scaled, scaler = modeling_functions.scale_features(X_train, X_test, selected_features)
                print(f"      Features scaled using StandardScaler.")

                # Modelltraining ohne GridSearch
                print(f"      Training model {model_name} with {fs_name} ({n_features} features) without tuning...")
                mae, mse, rmse, r2, residuals, y_pred = modeling_functions.train_model_without_tuning(model,X_train_scaled,y_train,X_test_scaled,y_test)
                print(
                    f"        Evaluation without tuning completed. MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                # Variablen zum Speichern der besten Ergebnisse initialisieren
                current_mse = mse
                current_model = model
                current_params = None  #
                if hasattr(model, 'get_params'):
                    current_params = model.get_params()  # Standardparameter holen, falls verfügbar
                current_mae = mae
                current_rmse = rmse
                current_r2 = r2
                current_y_pred = y_pred
                current_residuals = residuals

                #Prüfen ob enthalten in param_grid, wenn ja, Hyperparameter-Tuning durchführen
                if model_name in param_grid:
                    best_estimator, best_params, mae_tuned, mse_tuned, rmse_tuned, r2_tuned, residuals_tuned, y_pred_tuned = modeling_functions.train_model_with_gridsearch(model,X_train_scaled,y_train,X_test_scaled,y_test,param_grid[model_name])
                    print(f"        GridSearchCV completed for {model_name}. Best parameters: {best_params}")
                    print(
                        f"        Evaluation completed. MAE: {mae_tuned:.4f}, MSE: {mse_tuned:.4f}, RMSE: {rmse_tuned:.4f},"
                        f" R2: {r2_tuned:.4f}")

                    # Vergleich ohne/mit Tuning: Modell mit dem kleineren MSE behalten
                    if mse_tuned < current_mse:
                        current_mse = mse_tuned
                        current_model = best_estimator
                        current_params = best_params
                        current_mae = mae_tuned
                        current_rmse = rmse_tuned
                        current_r2 = r2_tuned
                        current_y_pred = y_pred_tuned
                        current_residuals = residuals_tuned
                        print(f"        GridSearch result chosen (better MSE).")
                    else:
                        print(f"        Standard model without tuning chosen (better MSE).")
                else:
                    print(f"        Model trained with standard hyperparameters.")

                # beste Kombination für jedes Modell anhand MSE auswählen
                if current_mse < best_score:

                    print(f"        New best score found for model {model_name}: MSE {current_mse:.4f}")
                    best_score = current_mse
                    best_fs_name = fs_name
                    best_n_features = n_features
                    best_model = current_model
                    best_scaler = scaler
                    best_selected_features = selected_features
                    best_predictions = y_pred
                    best_params[f"{model_name} with {fs_name}"] = current_params
                    best_mae = current_mae
                    best_rmse = current_rmse
                    best_r2 = current_r2
                    best_residuals = current_residuals

        # Modell Evaluation für jedes Modell speichern
        key = f"{model_name} with {best_fs_name} ({best_n_features} features)"
        results[key] = {
            'MAE': best_mae,
            'MSE': best_score,
            'RMSE': best_rmse,
            'R2': best_r2
        }
        print("Bestes Modell Metriken: ", results[key])
        best_estimators[key] = best_model
        best_scalers[key] = best_scaler
        best_selectors[key] = selector
        predictions_dict[key] = best_predictions
        residuals_dict[key] = best_residuals

        # Beste Kombination für jedes Modell speichern
        model_best_combinations[key] = {
            'fs_name': best_fs_name,
            'n_features': best_n_features,
            'selected_features': best_selected_features,
            'scaler': best_scaler,
            'best_model': best_model
        }

# Ergebnisse der Evaluation der Modelle als Dataframe speichern und ausgeben
results_evaluation = pd.DataFrame(results).T
print("\nErgebnisse der Modelle:\n", results_evaluation)

results_evaluation.to_csv('model_evaluation_results.csv', index=True)  # Speichern der Modellevaluation als CSV
# Hinzufügen der Vorhersagen der Modelle zu result_df
for model_name, predictions in predictions_dict.items():
    result_df[model_name + '_pred'] = predictions
print("\nDataFrame mit Vorhersagen:\n", result_df)

# Learning Curves der besten Kombinationen als Visualisierung ausgeben
learning_curves_data = {}
for model_name, combo in model_best_combinations.items():
    print(f"\nBeste Kombination für Modell {model_name}:")
    print(f"  Feature-Selection Methode: {combo['fs_name']}, Anzahl Features: {combo['n_features']}, Ausgewählte Features: {combo['selected_features']}")
    n_features = combo['n_features']
    selected_features = combo['selected_features']
    scaler = combo['scaler']
    best_model = combo['best_model']
    print("best_model: ", best_model)
    fs_name = combo['fs_name']


    #Trainings-/Validierungsfehler der trainierten Modelle
    X_train_best = scaler.transform(X_train[selected_features])
    y_train_best = y_train

    train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_best, y_train_best, cv=5,
                                                            scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, test_scores_mean, label='Validation error')
    plt.title(f'Learning Curve ({model_name} with {fs_name} - {n_features} features)')
    plt.xlabel('Training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()

    # Lernkurven speichern
    filename = f"learning_curve_{model_name}_{n_features}_features.png"
    modeling_functions.save_plot(filename)
    plt.show()

    # Speichern der Learning Curve-Daten in CSV
    learning_curves_data[f'{model_name}_train_sizes'] = train_sizes
    learning_curves_data[f'{model_name}_train_scores'] = train_scores_mean
    learning_curves_data[f'{model_name}_val_scores'] = test_scores_mean

learning_curves_df = pd.DataFrame(learning_curves_data)
learning_curves_df.to_csv('learning_curves_data.csv', index=False)  # Speichern der Learning Curve-Daten als CSV

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten von jedem Modell
plt.figure(figsize=(15, 10))
for model_name in predictions_dict.keys():
    plt.plot(result_df.index, result_df['sby_need'], label='Actual', color='blue')
    plt.plot(result_df.index, result_df[model_name + '_pred'], label='Predicted', color='grey')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.xlabel('Test Data Index')
    plt.ylabel('Anzahl benötigte Bereitschaftsfahrer (sby_need)')
    plt.legend()
    filename = f"predictions_vs_actual_{model_name}.png"
    modeling_functions.save_plot(filename)
    plt.show()

#Visualisierung der Residuen von jedem Modell
plt.figure(figsize=(15, 10))
for model_name, residuals in residuals_dict.items():
    plt.scatter(result_df.index, residuals, label='Residuals', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f'Residuals Plot for {model_name}')
    plt.xlabel('Test Data Index')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    filename = f"residuals_{model_name}.png"
    modeling_functions.save_plot(filename)
    plt.show()

#Visualisierung des MSE von jedem Modell
mse_values = [results[key]['MSE'] for key in results]
model_names = list(results.keys())
model_names = [name.split("with")[0].strip() for name in model_names]

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, mse_values, color='skyblue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom')

plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE of Each Best Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
modeling_functions.save_plot("MSE-Vergleich.png")
plt.show()


