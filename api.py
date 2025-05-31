import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from util import getJsonValue, setJsonValue#, checkCache
from polygon import RESTClient
import json 


try:
    import finnhub
    finnhubClient = finnhub.Client(api_key="d0pnqbhr01qgccua9u2gd0pnqbhr01qgccua9u30")
except Exception as ex:
    print("Finnhub client not available")
    print(ex)
    raise Exception("Finnhub client not available. Please check your API key or network connection.")

def checkLoadCache(cachePath):
    """
    Check if the cache exists for the given cacheName.
    If the cache exists it turns the cache, else it returns False.
    The cache can be in JSON or CSV format.
    """
    cacheName, extension = cachePath.rsplit(".", 1)
    try:
        with open(f"./tempDataCache/{cacheName}.{extension}", "r") as f:
            return json.load(f) if extension == "json" else pd.read_csv(f"./tempDataCache/{cacheName}.{extension}")
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Cache file {cacheName}.json is corrupted or empty.")
        return None
    
def saveCache(cachePath, data):
    """
    Save the data to the cache with the given cacheName.
    The data can be a dictionary (for JSON) or a DataFrame (for CSV).
    """
    cacheName, extension = cachePath.rsplit(".", 1)
    # print("TYPE OF DATA:", data)
    # print("EXTENSION IN SAVE CACHE:", extension)
    try:
        if extension == "json":
            with open(f"./tempDataCache/{cacheName}.json", "w") as f:###########Issue somehwere here
                json.dump(data, f, indent=4, default=str)  # Use default=str to handle non-serializable types
        elif extension == "csv":
            data.to_csv(f"./tempDataCache/{cacheName}.csv", index=None)  # Ensure data is a DataFrame
        else:
            raise ValueError("Unsupported file extension. Use 'json' or 'csv'.")
        print(f"Cache saved successfully to {cacheName}.{extension}")
        return True
    except Exception as ex:
        print(f"Error saving cache: {ex}")
        return False

def apiDecoratorFactory(apiservice, cachePath=None):#Copoilet Gen - I understand this conceptually, but not exactly sytatically
    """
        Factory decorator to check all API calls to ensure they are within the limits set in systemMetaData.json.
        If cacheName is provided, it will check if the cache exists and is not empty before making the API call.
        If no cacheName is provided it will not check or try to cache data (used for API's that do not require caching, like market status).

        Parameters:
        - apiservice: The name of the API service (e.g., "finnhub", "polygon").
        - cacheName: The name of the cache to check (optional). If provided, it will check if the cache exists and is not empty before making the API call.
        Returns:
        - A decorator that wraps the API function to check limits and cache.
    """
    def apiDecorator(func):#Copoilet Gen
        def wrapper(*args, **kwargs):
            # print("Name of API service:", apiservice)
            # print("Path IN decorator:", cachePath)
            # print("Cache Name:", cacheName)
            # print("Extension:", extension)
            ticker = None
            cacheName = None
            if cachePath != None: #If caccheName is provided,
                cacheName, extension = cachePath.rsplit(".", 1)
                ticker = args[0]
                cache =  checkLoadCache(f"{ticker}-{cachePath}")
                
                if cache is not None:
                    print(f"Cache for {cachePath} exists. Checking age of cache.")
                    cacheLastUpdated = getJsonValue("./Data/systemMetaData.json", "cacheLastUpdated")
                    print("Cache Last Updated:", cacheLastUpdated)
                    if datetime.now() - datetime.strptime(cacheLastUpdated, "%Y%m%d%H%M%S") > timedelta(days=1):
                        print("Cache is older than 1 day. Fetching new data from API...")
                        # Proceed to fetch new data
                    else:
                        print("Cache is less than 1 day old. Using cached data.")
                        return cache
                else:
                    print(f"Cache for {cachePath} does not exist or is empty. Fetching data from API...")
                    # Proceed to fetch new data

            if not confirmApiCall(apiservice):
                print(f"API call to {apiservice} is not allowed at this time.")
                return None
            try:
                funcOutput = func(*args, **kwargs)
                setJsonValue("./Data/systemMetaData.json", f"{apiservice}LastCall", str(datetime.now()))#Update the last call time
                setJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt", int(getJsonValue("./Data/systemMetaData.json", f"{apiservice}Curt")) + 1)#Increment the current count for this API service
            except Exception as ex:
                print(f"Error in API call to {apiservice} for ticker {ticker}: {ex}")
                return None
           
            if cacheName is not None: #args[0] shold be the the ticker symbol
                newCacheName = f"{args[0]}-{cacheName}"#Create a new cache name with the ticker symbol #-{datetime.now().strftime('%Y%m%d%H%M%S')}
                # print("New Cache Name:", newCacheName)
                saveCache(f"{newCacheName}.{extension}", funcOutput)
                setJsonValue("./Data/systemMetaData.json", "cacheLastUpdated", datetime.now().strftime("%Y%m%d%H%M%S"))#Update the last updated time of the cache
            
            return funcOutput
        return wrapper
    return apiDecorator

def minApiDecoratorFactory(apiservice, cachePath=None):
    def apiDecorator(func):
        def wrapper(*args, **kwargs):
            print("Decorator called with", args, kwargs)
            return func(*args, **kwargs)
        return wrapper
    return apiDecorator

def confirmApiCall(apiservice):#alphaVan, finnhub, and polygon
    limData = getJsonValue("./Data/systemMetaData.json", f"{apiservice}Limit").split(" ")
    lim, resetFreq = int(limData[0]), int(limData[2])
    lastCall = getJsonValue("./Data/systemMetaData.json", f"{apiservice}LastCall")
    
    if lim is None or lim == 0:
        print(f"API service \'{apiservice}\' not configured or limit not set.")
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

@apiDecoratorFactory("alphaVan", "allHistoricData.csv")
def getAllHistoric(ticker):
    """
        Gets all historic data for a given ticker from Alpha Vantage.
        Parameters:
        - ticker: Stock ticker symbol (e.g., "AAPL" for Apple Inc.)
        Returns:
        - dataframe containing the historic data for the ticker.
        
        Note: This function fetches daily data for the ticker, not intraday.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=full"
    r = requests.get(url)
    data = pd.read_csv(StringIO(r.text), index_col=None)
    # print(data.head())
    return data

@apiDecoratorFactory("finnhub", "quote.json")
def getLastKnownData(ticker):
    """
        Gets the last known data for a given ticker from Finnhub.
        Parameters:
        - ticker: Stock ticker symbol (e.g., "AAPL" for Apple Inc.)
         Returns:
        - DataFrame containing the last known data for the ticker.
        
        Note: This function fetches the most recent data point for the ticker.
    """
    finnhub_client = finnhub.Client(api_key="d0pnqbhr01qgccua9u2gd0pnqbhr01qgccua9u30")
    dataAsDict = finnhub_client.quote(ticker)
    df = pd.DataFrame([dataAsDict])#.drop(columns=["index"])
    return df #To get only the most recent ticker data

@apiDecoratorFactory("polygon", "SMA.json")
def getSMA(ticker, timespan="day", adjusted="true", window="50", series_type="close", order="desc"):
    """
    Get Simple Moving Average (SMA) for a given ticker.
    
    Parameters:
    - ticker: Stock ticker symbol
    - timespan: Time span for the SMA (e.g., "day", "week", "month")
    - adjusted: Whether to adjust for dividends and splits
    - window: The number of periods to calculate the SMA over
    - series_type: The type of price to use (e.g., "close", "open")
    - order: Order of the results ("asc" or "desc")
    
    Returns:
    - DataFrame containing the SMA data
    """
    client = RESTClient("ue0MRgNduDhjpt9DSsFSORpImHpqUITc")

    sma = client.get_sma(
        ticker="AAPL",
        timespan="day",
        adjusted="true",
        window="50",
        series_type="close",
        order="desc",
    )

    if hasattr(sma, "to_dict"):
        serializable = sma.to_dict()
    else:
        serializable = sma.__dict__

    return serializable



