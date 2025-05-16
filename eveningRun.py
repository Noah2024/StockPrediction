import sys
import os
import glob
import time
import json
from tensorflow.keras.models import load_model
from normalization import *
import numpy as np
import requests 
from io import StringIO
import ast

#!DO TO! Make Sure Trade data returns all 6 values needed DONE
#2) Compare prediction with stock data make decision about trade
#3) Add new trade to transaction history
#4) Considere a better way to store general info, cash, stock, and previousPredictions
#4) Set up Night Run to compare predictted with actual data
#5) Add a function to get the current stock data from the API
#6) Get an API to Trade stock and get information from the trade/HAve basic alternative
#7 Need to change normalize to accept a numpy array instead of a dataframe
#8 Need to update morning and evening run to use the new normalize function

def getCurtStockData(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
    r = requests.get(url)
    print(r)
    print(type(r))
    data = pd.read_csv(StringIO(r.text))
    return data[1:] #To get only the most recent ticker data

def getTransactionHistory(modelName):#Copoilet improved
    file_path = f'./ModelDataHistory/{modelName}.csv'
    # Load the entire CSV file, skipping the header
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Load all rows and columns
    print("Data Shape Before:", data.shape)

    # Ensure the data is always a 2D array
    data = np.atleast_2d(data)

    print("Data Shape After:", data.shape)
    return data[-1, :6]  # Get the last row and select the first 6 columns

def addTradeToQueue(modelName, Ticker, buySell, quantity, price, dateQue):
    tradeData = np.asarray([modelName, Ticker, buySell, quantity, price, dateQue]).reshape(1, -1)
    print("Trade Data:", tradeData)  # Debugging
    print("Data Type:", tradeData.dtype)  # Debugging
    with open(f"./Data/tradeQue.csv", "a", newline="") as file:
        np.savetxt(
            file,
            tradeData,
            delimiter=",",
            fmt="%s",  # Use string format specifier
        )

def Main():
    models = glob.glob("./ActiveModels/*.keras")
    loadedModels = {}
    for modelPath in models:
        modelName = os.path.basename(modelPath).split(".")[0]
        loadedModels[modelName] = load_model(modelPath)
        
    for modelName, model in loadedModels.items():
        print(f"MODEL NAME: {modelName} Loaded")
        modelMetaData = None
        with open(f"./ModelDataHistory/{modelName}.json", "r") as file:
            modelMetaData = json.load(file)
        cash = modelMetaData["cash"]
        shares = modelMetaData["shares"]
        normParams = ast.literal_eval(modelMetaData["normalParams"])

        #print("Cutrent Data: ", curtData)
        curtTradeData = getCurtStockData(modelMetaData["ticker"])#NOTE, THIS IS INEFFICENT, it transforms the entire dataset and not jus the last known row
        lastKnownDatapoint = curtTradeData.iloc[-1]
        curtTradeData = curtTradeData                           #But it won't let me get the last row without tranforming, so idk, efficency problem for later
        convertTsToEpoch(curtTradeData)
        normalizeAndUpdate(curtTradeData, normParams=normParams)
        
        # print("curtTradeData:", curtTradeData)
        # print("curtTradeData shape:", np.shape(curtTradeData))
        # print(curtTradeData.to_numpy())
        print("-----")
        dataToPredict = curtTradeData.to_numpy()[-1][np.newaxis, :]#Again, very innefficent, but whatever
        prediction = model.predict(dataToPredict)#Predict the stock data using the model
        predictedClose = denormalizeNp(prediction, normParams)[0][4]
        print(predictedClose)
        print(lastKnownDatapoint)
        print(curtTradeData)

        if (predictedClose - lastKnownDatapoint["close"])  < 0:
            print(f"{modelName} Predicts Stock will do Down, selling stock")
            cash += shares * curtTradeData.iloc[-1]["open"]
            shares = 0
        else:
            print(f"{modelName} Predicts Stock will do Up, buying stock")
            sharesToBuy = cash % lastKnownDatapoint["open"]#Get the current ammount of stock to buy
            cash -= sharesToBuy * lastKnownDatapoint["open"]
            shares += sharesToBuy

        print("Current Trade Data: ", curtTradeData)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        updateTradeData = np.append(lastKnownDatapoint, [cash, shares]).reshape(1, -1)
        print(" Update Trade Data:", updateTradeData)  
        with open(f"./ModelDataHistory/{modelName}.csv", "a", newline="") as file:
            np.savetxt(
                file,
                updateTradeData,
                delimiter=",",
                fmt="%s",
            )
        print(f"Cash: {cash}, Shares: {shares}")#Debugged with copoilet
        print("modelName type:", type(modelName))
        print("ticker type:", type(modelMetaData["ticker"]))
        print("buySell type:", type("Buy" if shares > 0 else "Sell"))
        print("shares type:", type(shares))
        print("curtTradeData[0] type:", type(lastKnownDatapoint["open"]))
        print("dateQue type:", type(time.strftime("%Y-%m-%d %H:%M:%S")))
        addTradeToQueue(
            str(modelName),  # Ensure modelName is a string
            str(modelMetaData["ticker"]),  # Ensure ticker is a string
            "Buy" if shares > 0 else "Sell",  # This is already a string
            str(shares),  # Convert shares to a string
            str(lastKnownDatapoint["open"]),  # Ensure curtTradeData[0] is a string
            str(time.strftime("%Y-%m-%d %H:%M:%S"))  # This is already a string
        )

        # print("TRADE DATA", curtTradeData)
        # print(f"Prediction: {prediction}")



if __name__ == "__main__":
    Main()