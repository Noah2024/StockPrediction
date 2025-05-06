import sys
import os
import glob
import numpy as np
from tensorflow.keras.models import load_model

#!DO TO! Make Sure Trade data returns all 6 values needed
#2) Compare prediction with stock data make decision about trade
#3) Add new trade to transaction history
#4) Considere a better way to store general info, cash, stock, and previousPredictions
#4) Set up Night Run to compare predictted with actual data
#5) Add a function to get the current stock data from the API
def getCurtStockData():
    return np.asarray([10,20,30,40,50,60])[np.newaxis, : ]#Example Data#API's will be added later

def getTransactionHistory(modelName):
    return np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1, usecols=(0,6))#
    pass

def Main():
    models = glob.glob("./ActiveModels/*.keras")
    curtData = getCurtStockData()
    loadedModels = {}
    for modelPath in models:
        modelName = os.path.basename(modelPath).split(".")[0]
        loadedModels[modelName] = load_model(modelPath)
        
    for modelName, model in loadedModels.items():
        print(f"MODEL NAME: {modelName} Loaded")
        prediction = model.predict(curtData)
        curtTradeData = getCurtTadeData(modelName)
        print("TRADE DATA", curtTradeData)
        print(f"Model: {modelPath} Prediction: {prediction}")



if __name__ == "__main__":
    Main()