import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    return df

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

# Отримання даних для кожного символу
quotes = []
for symbol in symbols:
    try:
        df = get_stock_data(symbol, start_date, end_date)
        quotes.append(df)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")

# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
opening_quotes = np.array([quote['Open'].values for quote in quotes]).astype(np.float64)
closing_quotes = np.array([quote['Close'].values for quote in quotes]).astype(np.float64)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
clustering = cluster.AffinityPropagation(random_state=0)
clustering.fit(edge_model.covariance_)
labels = clustering.labels_

# Виведення результатів кластеризації
num_labels = labels.max()
for i in range(num_labels + 1):
    print('Cluster ', i+1, '==>', ','.join(names[labels==i]))