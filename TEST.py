import requests
import pandas as pd
from io import StringIO

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=H3S85LY8M5OL60UU'
url = " https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
r = requests.get(url)
# data = r.json()
data = pd.read_csv(StringIO(r.text))
# print(data)
print(data.head)
print("----------")
Y = data[:-1].copy()  # everything except the last row
X = data[1:].copy()

print(X.head())
print("----------")
print(Y.head())
# print(X.head())
# print(y.head())