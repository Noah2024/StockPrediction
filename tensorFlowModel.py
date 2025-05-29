import json
import tensorflow as tf #-- must be imported before pd to work
from tensorflow import keras
from normalization import *
from api import *
import time

def tensorFlowModel(templatePath, modelName):
    print("Template Path:", templatePath)
    config = None
    print("Loading template data...")
    with open(templatePath, 'r') as file:
        config = json.load(file)
    print("Template data loaded successfully!")
    print("API Requesting...")
    data = None
    try: 
        data = getCompactDailyStock(config["ticker"])
    except Exception as e:
        print(f"Error in API Request: {e}")
        return None, None
    print("API Request Successful!")
    #------------------------------------
    print(data.head())
    lastRow = data[1:]#--Get the last row of the data# Used for first entry in model history file
    convertTsToEpoch(data)#--Converts timestamp string to epoch time
    normalizationParams = normalizeAndUpdate(data)
    print(data.head())

    yInputData = data[:-1].copy()  # everything except the last row
    xInputData = data[1:].copy()
    dataHeader = data.columns.tolist()
    
    finalActiveFunc = None if config["modelType"] == "regression" else 'softmax'
    print("Creating Model...")
    model = keras.Sequential()
    # ----------------------------------------------
    model.add(keras.layers.InputLayer(shape=xInputData.iloc[0].shape))  # Input layer expects data with shape `input_shape`
    for i in range(len(config["architecture"])-1):
        model.add(keras.layers.Dense(config["architecture"][i], activation=config["activation"]))
    model.add(keras.layers.Dense(yInputData.shape[1], activation=finalActiveFunc))
    model.compile(optimizer=config["optimizer"], loss=config["lossFunction"], metrics=config["metrics"])
    # data["timestamp"] = pd.to_datetime(data["Date"])
     #gives you the number of features (columns) in the dataset, which is what the Dense layer needs to know.

    print("Model Created")
    print("Fitting Model...")
    xInputAsNumpy = xInputData.to_numpy()
    yInputAsNumpy = yInputData.to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((xInputAsNumpy, yInputAsNumpy))
    # dataset = tf.data.Dataset.zip((xInputData.to_numpy(), yInputData.to_numpy()))#x and y are already pd.DataFrames so to use batchsize we need to zip them together
    dataset = dataset.batch(config["batchSize"]).prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(dataset, epochs=config["epochs"],batch_size=config["batchSize"], verbose=0)
    print("Model Fitted")

    modelData = lastRow.head(1).copy()
    modelData["Cash"] = 100
    modelData["Shares"] = 0

    model.save(config["modelPath"] + modelName + ".keras")
    print(modelData)
    print("Data Is", modelData)
    modelData.to_csv(f".\ModelDataHistory\{modelName}.csv", index = False)
    print("modelSaved")

    print("Saving Model MetaData")
    data = {
        "modelName": modelName,
        "ticker": config["ticker"],
        "cash": 100,
        "shares": 0,
        "normalParams": str(normalizationParams),
        "lastUpdated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    
    with open(f"./ModelDataHistory/{modelName}.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Model MetaData Saved")
    return model, history

def singlePredict():
        import ast
        from util import selectFile
        import os
        print("Enter the name of the path to the model you wish to predict")
        modelPath = selectFile()
        if not modelPath:
            print("No model selected.")
            return
        model = keras.models.load_model(modelPath)
        modelName = os.path.basename(modelPath)
        modelNoExtension = os.path.splitext(modelName)[0]
        modelMetaDataPath = f"./ModelDataHistory/{modelNoExtension}.json"

        if os.path.isfile(modelMetaDataPath):
            pass
        else:
            print("path to file does not exist - Exiting \"Single Predict\" ")
            return

        with open(modelMetaDataPath, "r") as file:
            modelMetaData = json.load(file)

        cash = modelMetaData["cash"]
        shares = modelMetaData["shares"]
        normParams = ast.literal_eval(modelMetaData["normalParams"])
        
        curtTradeData = getLastKnownData(modelMetaData["ticker"])#NOTE, THIS IS INEFFICENT, it transforms the entire dataset and not jus the last known row
        lastKnownDatapoint = curtTradeData
        curtTradeData["timestamp"] = int(time.time())
        # print(curtTradeData)
        convertTsToEpoch(curtTradeData)
        normalizeAndUpdate(curtTradeData, normParams=normParams)
        curtTradeData = curtTradeData.drop(labels=["latestDay","previousClose","change", "changePercent", "symbol"], axis=1)
        dataToPredict = curtTradeData.to_numpy()[-1][np.newaxis, :]#Again, very innefficent, but whatever
        print(dataToPredict)
        prediction = model.predict(dataToPredict)#Predict the stock data using the model
        print("Prediction", prediction)
        predictedClose = denormalizeNp(prediction, normParams)[0][4]
        print("Predicted Close for next day", predictedClose)
        print("Most Recent Trade Price", lastKnownDatapoint["open"], "Very off cause not normalized")
        if (predictedClose - lastKnownDatapoint["open"])  < 0:
            print(f"{modelName} Predicts Stock will do Down, selling stock")
            cash += shares * curtTradeData.iloc[-1]["open"]
            shares = 0
        else:
            print(f"{modelName} Predicts Stock will do Up, buying stock")
            sharesToBuy = cash % lastKnownDatapoint["open"]#Get the current ammount of stock to buy
            cash -= sharesToBuy * lastKnownDatapoint["open"]
            shares += sharesToBuy