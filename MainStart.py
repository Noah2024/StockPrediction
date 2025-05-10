
#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
import json
import os
import numpy as np
import time
import requests

print("Loading TensorFlow Dependencies...")
import tensorflow as tf #-- must be imported before pd to work
from tensorflow import keras
print("TensorFlow Keras loaded successfully!")
import pandas as pd
from io import StringIO

simStatus = False #Simulation Status

def yesNoInput(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def selectFile(): #Yoinked from chatGPT#Disabled for testing purposes
    current_file_path = os.path.abspath(__file__)
    return current_file_path
    while True:
        filePath = input("Please enter the full path to the file: (type \'exit\' to exit)")
        if os.path.isfile(filePath):
        # print(f"Selected file: {filePath}")
            return filePath
        elif filePath == "exit":
            print("Exiting file selection.")
            return None
        else:
            print("Invalid file path. Please try again.")

def convertTsToEpoch(df):#ChatGPT generated
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000_000

import pandas as pd

def normalizeAndUpdate(df: pd.DataFrame):
    """
    Normalize every numeric value in the DataFrame using min-max normalization.
    The original DataFrame is updated with the normalized values.
    
    Returns:
        - normalization_params (dict): Dictionary with column -> (min, max) for each column
    """
    normalization_params = {}

    # Loop through all numeric columns and normalize them
    for column in df.select_dtypes(include=["number"]).columns:
        col_min = df[column].min()
        col_max = df[column].max()
        normalization_params[column] = (col_min, col_max)
        # Normalize the column in place
        df[column] = (df[column] - col_min) / (col_max - col_min)

    return normalization_params

def denormalize(df: pd.DataFrame, normalization_params: dict):
    """
    Reverse min-max normalization using stored min and max values.
    
    Parameters:
        - df: The DataFrame with normalized values
        - normalization_params: Dictionary with column -> (min, max) for each column
    
    Returns:
        - denormalized_df (pd.DataFrame): The denormalized DataFrame
    """
    df = df.copy()

    for column, (col_min, col_max) in normalization_params.items():
        df[column] = df[column] * (col_max - col_min) + col_min
    
    return df

def SelectModelArchitecture():#Exit Not working yete
     while True:
        arch = input("Please enter model architecture (e.g., [64, 32, 16]): (type \'exit\' to exit)")
        try: 
            architecture = eval(arch)  # Convert string input to list
            if isinstance(architecture, list) and all(isinstance(i, int) for i in architecture):
                return architecture
            elif architecture == "exit":
                print("Exiting architecture selection.")
                return None
            else:
                print("Invalid architecture format. Please enter a list of integers.")
        except (SyntaxError, NameError):
            print("Invalid architecture format. Please enter a list of integers.")
    #inputShape, numClasses, architecture, 

def loadTemplateData(templatePath):
    dataHeader = None
    with open(templatePath, 'r') as file:
        data = json.load(file)
        print("PATH:", data["xInputPath"])
        xInputData = np.loadtxt(f'{data["xInputPath"]}', delimiter=',', skiprows=1)#
        yInputData = np.loadtxt(f'{data["yInputPath]"]}', delimiter=',', skiprows=1)
        dataHeader = np.loadtxt(f'{data["xInputPath"]}', delimiter=',', max_rows=1, dtype=str)  
    # Extracting the required fields from the loaded data
    
    return data, xInputData, yInputData

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
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={config['ticker']}&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
        r = requests.get(url)
        data = pd.read_csv(StringIO(r.text))
    except Exception as e:
        print(f"Error in API Request: {e}")
        return None, None
    print("API Request Successful!")
    #------------------------------------
    print(data.head())
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

    model.save(config["modelPath"] + modelName + ".keras")#Reshaped becuase by default it was populated with 1D data
    modelData = np.append(yInputAsNumpy[len(yInputAsNumpy)-1], [100, 0]).reshape(1, -1)
    print("Data Is", modelData)
    np.savetxt(
    f"./ModelDataHistory/{modelName}.csv",
    modelData,  # Append and reshape to a single row
    delimiter=",",
    fmt="%d",
    header=",".join(list(dataHeader) + ["cash", "shares"]),  # Convert header to a string and append new columns
    comments="")
    print("modelSaved")

    print("Saving Model MetaData")
    data = {
        "modelName": modelName,
        "ticker": config["ticker"],
        "cash": 100,
        "shares": 0,
        "lastUpdated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    
    with open(f"./ModelDataHistory/{modelName}.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Model MetaData Saved")
    return model, history
    return model, config

    
class MainStart:
    
    def __init__(self):
        self.models = []  # List to store models
        self.current_model = None  # Currently selected model

    def mainMenu(self):
        while True:
            print("-----------------------")
            print("Main Menu:")
            print(f"Simulation Status: {'Running' if simStatus else 'Not Running'}")
            print("-----------------------")
            print("1. Create New Model")
            print("2. Add Model To Simulation")
            print("3. Archive Model")
            print("4. Start Simulation")
            print("6. End Simulation")
            print("7. Check Current Results")
            print("8. Exit")

            choice = input("Enter your choice: ")

            if choice == '1':
                self.createNewModel()
            elif choice == '2':
                self.addFromArchive()
            elif choice == '3':
                self.archiveModel()
            elif choice == '4':
                self.run_simulation()
            elif choice == '5':
                self.check_results()
            elif choice == '6':
                self.check_results()
            elif choice == '7':
                self.checkResults()
            elif choice == '8':
                break
            else:
                print("Invalid choice, please try again.")
    
    def addFromArchive(Self):
        modelName = input("Please enter the name of the model to add: (type \'exit\' to exit)")
        if modelName == "exit":
            print("Exiting add model.")
            return None
        model = f"./ArchivedModels/{modelName}.keras"
        modelData = f"./ArchivedModels/{modelName}.json"
        modelHistory = f"./ArchivedModels/{modelName}.csv"
        if os.path.isfile(model) and os.path.isfile(modelData) and os.path.isfile(modelHistory):
            os.rename(model, f"./ActiveModels/{modelName}.keras")
            os.rename(modelData, f"./ModelDataHistory/{modelName}.json")
            os.rename(modelHistory, f"./ModelDataHistory/{modelName}.csv")
            print(f"Model {modelName} added successfully.")
        else:
            print(f"Model {modelName} not found in archived models.")
            return None

    def archiveModel(Self):#Disabled for testing purposes
        modelName = input("Please enter the name of the model to archive: (type \'exit\' to exit)")
        if modelName == "exit":
            print("Exiting archive model.")
            return None
        model = f"./ActiveModels/{modelName}.keras"
        modelData = f"./ModelDataHistory/{modelName}.json"
        modelHistory = f"./ModelDataHistory/{modelName}.csv"
        if os.path.isfile(model) and os.path.isfile(modelData) and os.path.isfile(modelHistory):
            os.rename(model, f"./ArchivedModels/{modelName}.keras")
            os.rename(modelData, f"./ArchivedModels/{modelName}.json")
            os.rename(modelHistory, f"./ArchivedModels/{modelName}.csv")
            print(f"Model {modelName} archived successfully.")
        else:
            print(f"Model {modelName} not found in active models.")
            return None
    

    def createNewModel(Self):
        modelName = input("Enter the name of the new model: ")
        default = yesNoInput("Do you want to load the default model template? (y/n): ")

        if default:
            print("Loading default model template...")
            model, config = tensorFlowModel("ModelTemplates/default.json", modelName)
            
            breakpoint()
        else:
           print("Enter the path to an alternative .json model template:")
           selectFile()
        return
    
    def checkResults(Self):
        import pandas as pd
        import matplotlib.pyplot as plt
        for file in os.listdir("./ModelDataHistory"):
            if file.endswith(".csv"):  
                print(f"Processing file: {file}")
                df = pd.read_csv("./ModelDataHistory/" + file)
                # Plot specific columns
                x = df.index
                plt.plot(x, df["<CLOSE>"], label="Open vs High")
                plt.xlabel("Index")
                plt.ylabel("Close Price")
                plt.title(f"Preformance of Model {file}")
                plt.legend()
                plt.show()
        
        

    
    def startSimulation(self):
        # Placeholder for starting a simulation
        if not self.models:
            print("No models available to run a simulation.")
            return
        print("Starting simulation with the following models:")
        for model in self.models:
            print(f"- {model}")
        # Simulate running the models (placeholder)
        print("Simulation started...")

MainMenu = MainStart()
MainMenu.mainMenu()
