
#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
import json
import os
import numpy as np
import time
import requests

from crontab import CronTab

print("Loading TensorFlow Dependencies...")
import tensorflow as tf #-- must be imported before pd to work
from tensorflow import keras
from normalization import *
from api import *
print("TensorFlow Keras loaded successfully!")

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

import pandas as pd

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
            #breakpoint()
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
        # Create a cron object for the current user
        cron = CronTab(user=True)

        # Add a new job
        job = cron.new(command='python3 /path/to/your_script.py', comment='my_automatic_task')

        # Set schedule (daily at 7 AM)
        job.setall('0 7 * * *')

        # Write the job to the crontab
        cron.write()

MainMenu = MainStart()
MainMenu.mainMenu()
