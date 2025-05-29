import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from util import getJsonValue, setJsonValue

try:
    import finnhub
    finnhubClient = finnhub.Client(api_key="d0pnqbhr01qgccua9u2gd0pnqbhr01qgccua9u30")
except Exception as ex:
    print("Finnhub client not available")
    print(ex)
    raise Exception("Finnhub client not available. Please check your API key or network connection.")

def apiDecoratorFactory(apiservice):#Copoilet Gen#I understand this conceptually, but not exactly sytatically
    def apiDecorator(func):#Copoilet Gen
        def wrapper(*args, **kwargs):
            # print("Name of API service:", apiservice)
            if not confirmApiCall(apiservice):
                print(f"API call to {apiservice} is not allowed at this time.")
                return None
            funcOutput = func(*args, **kwargs)
            setJsonValue("./Data/systemMetaData.json", f"{apiservice}LastCall", str(datetime.now()))#Update the last call time
            setJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt", int(getJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt")) + 1)#Increment the current count for this API service
            return 
        return wrapper
    return apiDecorator

def confirmApiCall(apiservice):#alphaVan, finnhub, and polygon
    limData = getJsonValue("./Data/systemMetaData.json", f"{apiservice}Limit").split(" ")
    lim, resetFreq = int(limData[0]), int(limData[2])
    lastCall = getJsonValue("./Data/systemMetaData.json", f"{apiservice}LastCall")
    
    if lim is None or lim == 0:
        print(f"API service {apiservice} not configured or limit not set.")
        return False

    if datetime.now() - datetime.strptime(lastCall, "%Y-%m-%d %H:%M:%S.%f") > timedelta(minutes=resetFreq):#Line gen with Copilot
        print(f"API service {apiservice} limit reset. Last call was at {lastCall}.")
        setJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt", 0)
        return True

    currentCount = int(getJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt")) + 1 # Increment current count for this call
    if currentCount > lim:
        print(f"API service {apiservice} limit reached. Current count: {currentCount}, Limit: {lim}")
        return False
    
    return True
    
@apiDecoratorFactory("finnhub")
def isMarketOpen():
    status = finnhubClient.market_status(exchange='US')["isOpen"]
    rtn = "closed" if status == False else "open"
    return rtn

def getAllHistoric(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
    r = requests.get(url)
    data = pd.read_csv(StringIO(r.text), index_col=None)
    return data

def getLastKnownData(ticker):
    finnhub_client = finnhub.Client(api_key="d0pnqbhr01qgccua9u2gd0pnqbhr01qgccua9u30")
    dataAsDict = finnhub_client.quote(ticker)
    df = pd.DataFrame([dataAsDict])#.drop(columns=["index"])
    return df #To get only the most recent ticker data