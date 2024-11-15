""" Exemplarische GUI für die Anzeige der benötigten Bereitschaftsfaherer erstellen"""

import pandas as pd
import tkinter as tk
from tkinter import ttk
import mplcursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


future_data = pd.read_csv('predictions.csv')
future_data['date'] = pd.to_datetime(future_data['date'], format='%Y-%m-%d')
future_data.set_index('date', inplace=True)


# Funktionen zur Diagrammerstellung
def plot_daily(ax, future_data):
    ax.clear()
    ax.plot(future_data.index, future_data['predicted_sby_need_adjusted'], label='vorhergesagte benötigte Bereitschaftsfahrer', color='green', marker='o')
    ax.set_title('Anzahl benötigte Bereitschaftsfahrer pro Tag')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Anzahl Bereitschaftsfahrer')
    ax.legend()

def plot_weekly(ax, future_data):
    ax.clear()
    weekly_data = future_data['predicted_sby_need_adjusted'].resample('W').sum().astype(int)
    weekly_data = weekly_data.to_frame(name='predicted_sby_need_adjusted')
    weekly_data['week'] = weekly_data.index.isocalendar().week

    ax.plot(weekly_data.index, weekly_data['predicted_sby_need_adjusted'], label='vorhergesagte benötigte Bereitschaftsfahrer', color='green', marker='o')
    ax.set_title('Anzahl benötigte Bereitschaftsfahrer pro Woche')
    ax.set_xlabel('Kalenderwoche')
    ax.set_ylabel('Anzahl Bereitschaftsfahrer')
    ax.legend()
    ax.set_xticks(weekly_data.index)
    ax.set_xticklabels([f"KW {int(wk)}" for wk in weekly_data['week']], rotation=45, ha='center')  # Zentrierte Labels

# Funktion zur Erstellung der Tabellen
def create_table(frame, future_data, freq):
    for widget in frame.winfo_children():
        widget.destroy()

    if freq == 'daily':
        table_data = future_data[['predicted_sby_need_adjusted']].astype(int)
        headers = ('Tag', 'Benötigte Bereitschaftsfahrer')
    elif freq == 'weekly':
        table_data = future_data[['predicted_sby_need_adjusted']].resample('W').sum().astype(int)
        table_data['week'] = table_data.index.isocalendar().week
        headers = ('Woche', 'Benötigte Bereitschaftsfahrer')

    table = ttk.Treeview(frame, columns=headers, show='headings')
    table.heading(headers[0], text=headers[0], anchor='center')
    table.heading(headers[1], text=headers[1], anchor='center')
    table.column(headers[0], anchor='center')
    table.column(headers[1], anchor='center')
    table.column(headers[0], anchor='center', width=50)  # Spaltenbreite angepasst
    table.column(headers[1], anchor='center', width=150)  # Spaltenbreite angepasst

    for index, row in table_data.iterrows():
        if freq == 'daily':
            table.insert('', 'end', values=(index.strftime('%Y-%m-%d'), row['predicted_sby_need_adjusted']))
        elif freq == 'weekly':
            table.insert('', 'end', values=(f"KW {int(row['week'])}", row['predicted_sby_need_adjusted']))

    table.pack(fill='both', expand=True)

# Hauptanwendung erstellen
def create_app(future_data):
    root = tk.Tk()
    root.title("Bereitschaftsdienst Vorhersagen")

    tabControl = ttk.Notebook(root)
    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)
    tabControl.add(tab1, text='Täglich')
    tabControl.add(tab2, text='Wöchentlich')
    tabControl.pack(expand=1, fill="both")

    # Tagesansicht
    frame1 = ttk.Frame(tab1)
    frame1.pack(fill='both', expand=True)
    fig1 = Figure(figsize=(12, 5), dpi=100)
    ax1 = fig1.add_subplot(111)
    plot_daily(ax1, future_data)
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.get_tk_widget().pack()
    canvas1.draw()

    # mplcursors für die Tagesansicht explizit verbinden
    mplcursors.cursor(ax1, hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Date: {future_data.index[int(min(sel.index, len(future_data) - 1))].strftime('%Y-%m-%d')}\nPredicted: {future_data['predicted_sby_need_adjusted'][int(min(sel.index, len(future_data) - 1))]}"))

    table_frame1 = ttk.Frame(tab1)
    table_frame1.pack(fill='both', expand=True)
    create_table(table_frame1, future_data, 'daily')

    # Wochenansicht
    frame2 = ttk.Frame(tab2)
    frame2.pack(fill='both', expand=True)
    fig2 = Figure(figsize=(12, 5), dpi=100)
    ax2 = fig2.add_subplot(111)
    plot_weekly(ax2, future_data)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.get_tk_widget().pack()
    canvas2.draw()

    # mplcursors für die Wochenansicht explizit verbinden
    weekly_data = future_data['predicted_sby_need_adjusted'].resample('W').sum().astype(int).to_frame(name='predicted_sby_need_adjusted')
    weekly_data['week'] = weekly_data.index.isocalendar().week

    mplcursors.cursor(ax2, hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Week: KW {weekly_data['week'][int(min(sel.index, len(weekly_data) - 1))]}\nSummed Predicted: {weekly_data['predicted_sby_need_adjusted'][int(min(sel.index, len(weekly_data) - 1))]}"))

    table_frame2 = ttk.Frame(tab2)
    table_frame2.pack(fill='both', expand=True)
    create_table(table_frame2, future_data, 'weekly')

    root.mainloop()

# Anwendung starten
if __name__ == "__main__":
    create_app(future_data)
