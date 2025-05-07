import sys
import os
import glob
import numpy as np
from tensorflow.keras.models import load_model

#!DO TO! Make Sure Trade data returns all 6 values needed DONE
#2) Compare prediction with stock data make decision about trade
#3) Add new trade to transaction history
#4) Considere a better way to store general info, cash, stock, and previousPredictions
#4) Set up Night Run to compare predictted with actual data
#5) Add a function to get the current stock data from the API
def getCurtStockData():
    return np.asarray([24.3,22.3,23.6,24.4,1000,0])[np.newaxis, : ]#Example Data#API's will be added later

def getTransactionHistory(modelName):
    # print("Test: ", np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1))
    header = np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=",", max_rows=1, dtype=str)
    cash, shares = np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1, usecols=(6,7))#Get the cash and shares from the transaction history
    return cash, shares,header, np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1, usecols=(0,1,2,3,4,5))#
    pass

def Main():
    models = glob.glob("./ActiveModels/*.keras")
    #curtData = getCurtStockData() There is no current data cause the market has just opened
    loadedModels = {}
    for modelPath in models:
        modelName = os.path.basename(modelPath).split(".")[0]
        loadedModels[modelName] = load_model(modelPath)
        
    for modelName, model in loadedModels.items():
        print(f"MODEL NAME: {modelName} Loaded")
        #print("Cutrent Data: ", curtData)
        cash, shares, header, curtTradeData = getTransactionHistory(modelName)
        prediction = model.predict(curtTradeData[np.newaxis, :])#Predict the stock data using the model
        
        if (prediction[0][0] - curtTradeData[0])  < 0:
            print(f"{modelName} Predicts Stock will do Down, selling stock")
            cash += shares * curtTradeData[0]
            shares = 0
        else:
            print(f"{modelName} Predicts Stock will do Up, buying stock")
            sharesToBuy = cash % curtTradeData[0]#Get the current ammount of stock to buy
            cash -= sharesToBuy * curtTradeData[0]
            shares += sharesToBuy
        print("Current Trade Data: ", curtTradeData)
        updateTradeData = np.append(curtTradeData, [cash, shares]).reshape(1, -1)
        print(" Update Trade Data:", updateTradeData)       
        np.savetxt(
            f"./FakeTranscationHistory/{modelName}.csv",
            updateTradeData,
            delimiter=",",
            fmt="%d",
            header=",".join(header)  # Convert the NumPy array to a comma-separated stringer
        )
        print(f"Cash: {cash}, Shares: {shares}")


        # print("TRADE DATA", curtTradeData)
        # print(f"Prediction: {prediction}")



if __name__ == "__main__":
    Main()