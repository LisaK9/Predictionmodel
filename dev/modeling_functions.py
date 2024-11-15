import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import pickle
import matplotlib.pyplot as plt


def load_and_preprocess_data(filepath):
    """
        Ladet die CSV-Datei mit den Daten für das Modelltraining, wandelt das Datum um und
        setzt es als Index. Da der Bereitschaftsplan immer am 15. eines Monats zur Verfügung stehen muss,
        werden nur Daten bis zum 14. eingeschlossen, mit den Daten bis zum Vortag trainiert werden soll.
        Damit das Modell und die Vorhersagen präzise evaluiert werden kann, werden die vorhandenen Daten
        des Folgemonats als Testdaten für das Modell, bzw. später als Input für die Vorhersage verwendet.
        Args:
        filepath (str): Der Pfad zur CSV-Datei.

        Returns:
        pd.DataFrame: DataFrame mit den Trainingsdaten gefiltert bis zum 14.04.2019.

        Nach der Inbetriebnahme werden die Daten dann wie folgt geladen:

        data = pd.read_csv(filepath)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        max_date = data.index.max() # Max-Datum bestimmen
        max_date_limit = max_date.replace(day=14)  # Setze den Tag auf den 14. Dies dient ausschließlich zur Sicherheit, normalerweise sollte das Max-Datum ausreichen

        filtered_data = data[data.index <= max_date_limit] # Daten bis zum 14. des Max-Datums filtern

        return filtered_data

        """
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day

    # Funktion zur Berechnung der Woche im Monat
    def week_of_month(dt):
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return (adjusted_dom - 1) // 7 + 1

    # Neues Feature 'week_of_month' hinzufügen
    data['week_of_month'] = data['date'].apply(week_of_month)
    data.set_index('date', inplace=True)
    return data[:'2019-04-14']


# Features und Zielvariable festlegen
def define_features_and_target(data):
    """
    Hier werden die Features und die Zielvariable, die für das Modelltraining verwendet werden,
       festgelegt.
    Args:
        data (pd.DataFrame): DataFrame mit den Trainingsdaten.

    Returns:
        data[features] (pd.DataFrame), enthält die Daten der Features
        data[target] (pd.Series), enthält die Daten der Zielvariable
        features (List), enthält eine Liste der verwendeten Features.
       """
    features = ['n_duty', 'month', 'day_of_week', 'season', 'week', 'holiday', 'year', 'n_sick', 'day', 'week_of_month', 'calls'
                ]
    target = 'sby_need'
    return data[features], data[target], features

# Train-Test-Split durchführen
def split_data(X, y):
    """
       Führt die Aufteilung der Daten in Trainings- und Testdaten durch.

    Args:
        X (pd.DataFrame): Daten der Features.
        y (pd.Series): Daten der Zielvariable.

    Returns:
        Trainings- und Testdaten (X_train, X_test, y_train, y_test).
       """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def define_feature_selectors():
    """
       Hier werden die Feature-Selektionmethoden definiert, die verwendet werden sollen.
       Returns:
        Dictionary mit den Namen der Selektoren und den entsprechenden Instanzen
       """
    feature_selectors = {
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    return feature_selectors



def define_models_and_params():
    """
       Definiert die Modelle und Hyperparameter-Raster für das Modelltraining.

    Returns:
        models: Dictionary der Modelle und dem
        param_grid: Dictionary der Hyperparameter-Raster
       """
    models = {
        #"Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        "LightGBM Regressor": LGBMRegressor(random_state=42),
        "CatBoost Regressor": CatBoostRegressor(logging_level='Silent', random_state=42)
    }

    param_grid = {
        "Random Forest Regressor": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5]
        },
        "Gradient Boosting Regressor": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        },
        "XGBoost Regressor": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        },
        "LightGBM Regressor": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100]
        },
        "CatBoost Regressor": {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [3, 4, 5]
        }
    }
    return models, param_grid



def feature_selection(fs_name, selector, X_train, y_train, n_features):
    """
    Führt die Feature-Auswahl basierend auf der Feature-Selektionstechnik durch.
    Args:
        fs_name (str): Der Name der Feature-Selektionstechnik.
        selector (object): Der Selektor, der verwendet wird.
        X_train (pd.DataFrame): Die Trainingsdaten.
        y_train (pd.Series): Die Zielvariable.
        n_features (int): Die Anzahl der auszuwählenden Features.

    Returns:
        np.ndarray: Indizes der ausgewählten Features.
    """
    if fs_name == 'Random Forest':
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_feature_indices = indices[:n_features]
    else:
        print(f"  Warning: Unknown feature selector {fs_name}, using default method.")
        selector.fit(X_train, y_train)
        selected_feature_indices = selector.get_support(indices=True)

    return selected_feature_indices


def scale_features(X_train, X_test, selected_features):
    """
    Skaliert die ausgewählten Features des Trainings- und Test-Sets.
    Args:
        X_train (pd.DataFrame): Die Trainingsdaten.
        X_test (pd.DataFrame): Die Testdaten.
        selected_features (list): Liste der auszuwählenden Features.

    Returns:
        X_train_scaled: skalierte Trainingsdaten
        X_test_scaled: skalierte Testdaten
        scaler: verwendeter Scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])
    return X_train_scaled, X_test_scaled, scaler


def train_model_without_tuning(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trainiert das Modell ohne Hyperparameter-Tuning und berechnet die Werte zur Evaluierung.

    Args:
        model (object): Das zu trainierende Modell.
        X_train_scaled (np.ndarray): Die skalierten Trainingsdaten.
        y_train (pd.Series): Die Zielvariable für die Trainingsdaten.
        X_test_scaled (np.ndarray): Die skalierten Testdaten.
        y_test (pd.Series): Die Zielvariable für die Testdaten.

    Returns:
        MAE, MSE, RMSE, R², Residuen und Vorhersagen
    """
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    return mae, mse, rmse, r2, residuals, y_pred


def train_model_with_gridsearch(model, X_train_scaled, y_train, X_test_scaled, y_test, param_grid):
    """
    Trainiert das Modell mit GridSearch und berechnet die Werte für die Evaluierung.

    Args:
        model (object): Das zu trainierende Modell.
        X_train_scaled (np.ndarray): Die skalierten Trainingsdaten.
        y_train (pd.Series): Die Zielvariable für die Trainingsdaten.
        X_test_scaled (np.ndarray): Die skalierten Testdaten.
        y_test (pd.Series): Die Zielvariable für die Testdaten.
        param_grid (dict): Das Hyperparameter-Raster für GridSearch.

    Returns:
        bestes Modell, beste Hyperparameter, MAE, MSE, RMSE, R², Residuen und Vorhersagen.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    y_pred_tuned = grid_search.best_estimator_.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_tuned)
    mse = mean_squared_error(y_test, y_pred_tuned)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_tuned)
    residuals = y_test - y_pred_tuned
    return grid_search.best_estimator_, grid_search.best_params_, mae, mse, rmse, r2, residuals, y_pred_tuned


def save_model(model, filename):
    """
    Speichert ein Modell in einer Pickle-Datei.

    Args:
    model (object): Das zu speichernde Objekt.
    filename (str): Der Name der Datei, in der das Objekt gespeichert wird.

    Returns:
    None
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
            print(f"Model saved as {filename}")
    except Exception as e:
        print(f"Error saving the model: {e}")


def save_plot(filename, dpi=300):
    """
    Speichert die aktuelle Matplotlib-Grafik als Datei.

    Args:
    filename (str): Der Name der Datei.
    dpi (int): Die Auflösung der Grafik. Standardmäßig 300 DPI.

    Returns:
    None
    """
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
