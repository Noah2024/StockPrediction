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
    cash, shares = np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1, usecols=(6,7))#Get the cash and shares from the transaction history
    return cash, shares, np.loadtxt(f'./FakeTranscationHistory/{modelName}.csv', delimiter=',',skiprows=1, usecols=(0,1,2,3,4,5))#
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
        print("Cutrent Data: ", curtData)
        cash, shares, curtTradeData = getTransactionHistory(modelName)
        #Lines 34 to 41 are used as a benchmark to scale buying and selling, may be a bad idea, may remove later
        accuracyPrediction = model.predict(curtTradeData[np.newaxis, :])#Predict the stock data using the model
        
        diff = curtTradeData[0] - accuracyPrediction[0][0]#Get the difference between the predicted and actual data
        accuracy = (1 - abs(diff / curtTradeData[0]))#* 100 
        # print(f"Actual: {curtTradeData[0]}, Predicted: {accuracyPrediction[0][0]}")
        # print(f"Difference: {diff}")
        # print(f"Accuracy: {accuracy}%")
        #[np.newaxis, :]
        prediction = model.predict(curtData)#Predict the stock data using the model
        # print(f"Actual: {curtData[0][0]}, Predicted: {prediction[0][0]}")
        # print(f"Actual NEXT STEP Prediction: {prediction}")
        # print("Actual Diff: ", prediction[0][0] - prediction[0][0])
        if (prediction[0][0] - curtData[0][0])  < 0:
            print(f"{modelName} Predicts Stock will do Down, selling stock")
            cash += shares * curtData[0][0]
            shares = 0
        else:
            print(f"{modelName} Predicts Stock will do Up, buying stock")
            sharesToBuy = cash % curtData[0][0]#Get the current ammount of stock to buy
            cash -= sharesToBuy * curtData[0][0]
            shares += sharesToBuy
        print(f"Cash: {cash}, Shares: {shares}")

        
        # print("TRADE DATA", curtTradeData)
        # print(f"Prediction: {prediction}")



if __name__ == "__main__":
    Main()