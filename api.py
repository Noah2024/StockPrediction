import requests
import pandas as pd
from io import StringIO

def isMarketOpen():
    import requests
    url = 'https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=demo'
    r = requests.get(url)
    data = r.json()
    usStatus = None
    for market in data["markets"]:
        if market["region"] == "United States":
            usStatus = market["current_status"]
            break
    print("US Market Status:", usStatus)
    return usStatus

def getCompactDailyStock(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
    r = requests.get(url)
    data = pd.read_csv(StringIO(r.text), index_col=None)
    return data

def getLastKnownData(ticker):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey=H3S85LY8M5OL60UU&datatype=csv'
    r = requests.get(url)
    data = pd.read_csv(StringIO(r.text))
    print("Last Known Data", data)
    return data #To get only the most recent ticker data