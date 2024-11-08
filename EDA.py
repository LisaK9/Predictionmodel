import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

#Daten einlesen
data=pd.read_csv("sickness_table.csv")
print(data.head(5))
# Grundlegende Informationen und Statistiken anzeigen
print("Grundlegende Informationen:")
print(data.info())

print("\nFehlende Werte pro Spalte:")
print(data.isnull().sum())

#Anzahl der Zeilen und Spalten (Datenqualität)
print(f"Zeilen und Spalten: {data.shape}")

#Eindeutige Werte im Datensatz (Datenqualität)
print("Eindeutige Werte im Datensatz:")
print(data.nunique())

# Konvertierung der 'date'-Spalte in das Datumsformat
data['date'] = pd.to_datetime(data['date'])

#Indexspalte löschen
data = data.drop(columns=["Unnamed: 0"])

data['activation_rate'] = (data['sby_need']/data['n_sby'])*100

print("\nBeschreibende Statistiken:")
print(data.describe())
data = data.drop(columns=["n_sby"])
# zusätzliche Features
data['day_of_week'] = data['date'].dt.dayofweek
data['week'] = data['date'].dt.isocalendar().week
data['month'] = data['date'].dt.month
data['season'] = data['month'] % 12 // 3 + 1  # 1: Winter, 2: Frühling, 3: Sommer, 4: Herbst
data['year'] = data['date'].dt.year

#Feiertage
de_holidays = holidays.Germany()
#Binäres Feature, ob Feiertag oder nicht
data['holiday'] = data['date'].apply(lambda x: 1 if x in de_holidays else 0)

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Ändern des Datentyps zu int64
data['week'] = data['week'].astype('int64')
data['day_of_week'] = data['day_of_week'].astype('int64')
data['month'] = data['month'].astype('int64')
data['season'] = data['season'].astype('int64')
data['year'] = data['year'].astype('int64')
data['calls'] = data['calls'].astype('int64')
data['sby_need'] = data['sby_need'].astype('int64')
data['dafted'] = data['dafted'].astype('int64')
print('Angepasste Datentypen: ',data.info())

# Plotten der Zeitreihen für die relevanten Variablen
fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# Plot für n_sick
axs[0].plot(data.index, data['n_sick'], color='blue')
axs[0].set_title('n_sick (Anzahl der Krankenmeldungen)')
axs[0].set_ylabel('n_sick')

# Plot für calls
axs[1].plot(data.index, data['calls'], color='orange')
axs[1].set_title('calls (Anzahl der Anrufe)')
axs[1].set_ylabel('calls')

# Plot für n_duty
axs[2].plot(data.index, data['n_duty'], color='green')
axs[2].set_title('n_duty (Anzahl der Mitarbeiter im Dienst)')
axs[2].set_ylabel('n_duty')

# Plot für sby_need
axs[3].plot(data.index, data['dafted'], color='yellow')
axs[3].set_title('dafted')
axs[3].set_ylabel('dafted')
axs[3].set_xlabel('Datum')

# Plot für sby_need
axs[4].plot(data.index, data['sby_need'], color='red')
axs[4].set_title('sby_need (Zielvariable)')
axs[4].set_ylabel('sby_need')
axs[4].set_xlabel('Datum')

plt.tight_layout()
plt.show()


# Korrelationen zwischen den numerischen Variablen
correlation_matrix = data.corr()

# Heatmap der Korrelationen
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix')
plt.show()
print(data.dtypes)

# For analysis, select relevant numerical columns for correlation matrix and scatterplot
numerical_columns = ['n_sick', 'calls', 'n_duty', 'sby_need', 'dafted']

# Adjust the overall context for the plot to reduce font sizes
#sns.set_context("talk", font_scale=0.2)  # Adjust font_scale to make labels smaller

# Create pairplot with smaller labels and distributions on the diagonal
g = sns.pairplot(
    data[numerical_columns],
    diag_kind="kde",          # Use kernel density estimate for the diagonal plots
)

# Show the plot
plt.show()

# Select relevant columns for seasonal decomposition
variables = ['calls', 'n_sick', 'sby_need']

# Perform seasonal decomposition for each variable
decompositions = {}
for var in variables:
    decompositions[var] = seasonal_decompose(data[var], model='additive', period=30)

# Plot seasonal decompositions
for var, decomposition in decompositions.items():
    plt.figure(figsize=(10, 8))
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition of {var}', y=1.02)
plt.show()

# Extract month and weekday from the index for analysis
data['month'] = data.index.month
data['weekday'] = data.index.weekday

# Calculate average values per month and weekday for each variable
heatmap_data_calls = data.pivot_table(values='calls', index='month', columns='weekday', aggfunc=np.mean)
heatmap_data_n_sick = data.pivot_table(values='n_sick', index='month', columns='weekday', aggfunc=np.mean)
heatmap_data_sby_need = data.pivot_table(values='sby_need', index='month', columns='weekday', aggfunc=np.mean)

# Separate Heatmaps plotten
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data_calls, cmap='YlOrBr', annot=True, fmt=".1f", cbar=True)
plt.title('Seasonality Heatmap of Calls')
plt.xlabel('Weekday')
plt.ylabel('Month')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data_n_sick, cmap='YlOrBr', annot=True, fmt=".1f", cbar=True)
plt.title('Seasonality Heatmap of n_sick')
plt.xlabel('Weekday')
plt.ylabel('Month')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data_sby_need, cmap='YlOrBr', annot=True, fmt=".1f", cbar=True)
plt.title('Seasonality Heatmap of sby_need')
plt.xlabel('Weekday')
plt.ylabel('Month')
plt.show()

# Trend analysis per season
# Define seasons based on months
data['season'] = np.where(data['month'].isin([12, 1, 2]), 'Wi',
                          np.where(data['month'].isin([3, 4, 5]), 'Fr',
                                   np.where(data['month'].isin([6, 7, 8]), 'So', 'He')))

# Calculate average trend per season
seasonal_trends = data.groupby(['season', data.index.year]).mean()[['calls', 'n_sick', 'sby_need']]

# Plotting trend analysis per season
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
variables = ['calls', 'n_sick', 'sby_need']

for i, var in enumerate(variables):
    seasonal_trends[var].unstack(level=0).plot(ax=axes[i], marker='o')
    axes[i].set_title(f'Trend Analysis of {var} by Season')
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel(var)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


monthly_avg = data.groupby(data.index.month)[['calls', 'n_sick', 'sby_need']].mean()
monthly_avg.index.name = 'Month'

# Durchschnitt pro Wochentag
weekday_avg = data.groupby(data.index.weekday)[['calls', 'n_sick', 'sby_need']].mean()
weekday_avg.index = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

# Individual plots for each variable per month
monthly_avg['calls'].plot(ax=axes[0, 0], color='b', marker='o')
axes[0, 0].set_title('Monthly Average of Calls')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Calls')

monthly_avg['n_sick'].plot(ax=axes[0, 1], color='r', marker='o')
axes[0, 1].set_title('Monthly Average of n_sick')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('n_sick')

monthly_avg['sby_need'].plot(ax=axes[0, 2], color='g', marker='o')
axes[0, 2].set_title('Monthly Average of sby_need')
axes[0, 2].set_xlabel('Month')
axes[0, 2].set_ylabel('sby_need')

# Individual plots for each variable per weekday
weekday_avg['calls'].plot(ax=axes[1, 0], color='b', marker='o')
axes[1, 0].set_title('Weekday Average of Calls')
axes[1, 0].set_xlabel('Weekday')
axes[1, 0].set_ylabel('Calls')

weekday_avg['n_sick'].plot(ax=axes[1, 1], color='r', marker='o')
axes[1, 1].set_title('Weekday Average of n_sick')
axes[1, 1].set_xlabel('Weekday')
axes[1, 1].set_ylabel('n_sick')

weekday_avg['sby_need'].plot(ax=axes[1, 2], color='g', marker='o')
axes[1, 2].set_title('Weekday Average of sby_need')
axes[1, 2].set_xlabel('Weekday')
axes[1, 2].set_ylabel('sby_need')

plt.tight_layout()
plt.show()