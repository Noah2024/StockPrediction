
#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
import tensorflow as tf
from tensorflow import keras
import json
import os
import numpy as np
import time

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
            print(f"Selected file: {filePath}")
            return filePath
        elif filePath == "exit":
            print("Exiting file selection.")
            return None
        else:
            print("Invalid file path. Please try again.")

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
        print("PATH:", config["xInputPath"])
        xInputData = np.loadtxt(f'{config["xInputPath"]}', delimiter=',', skiprows=1)#
        yInputData = np.loadtxt(f'{config["yInputPath"]}', delimiter=',', skiprows=1)
        dataHeader = np.loadtxt(f'{config["xInputPath"]}', delimiter=',', max_rows=1, dtype=str)

    finalActiveFunc = None if config["modelType"] == "regression" else 'softmax'

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=xInputData[0].shape))  # Input layer expects data with shape `input_shape`
    for i in range(len(config["architecture"])-1):
        model.add(keras.layers.Dense(config["architecture"][i], activation=config["activation"]))
    model.add(keras.layers.Dense(config["numClasses"], activation=finalActiveFunc))
    model.compile(optimizer=config["optimizer"], loss=config["lossFunction"], metrics=config["metrics"])
    
    print("Model Created")
    print("Fitting Model...")
    dataset = tf.data.Dataset.from_tensor_slices((xInputData, yInputData))
    dataset = dataset.batch(config["batchSize"]).prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(dataset, epochs=config["epochs"],batch_size=config["batchSize"], verbose=0)
    print("Model Fitted")

    model.save(config["modelPath"] + modelName + ".keras")#Reshaped becuase by default it was populated with 1D data
    np.savetxt(
    f"./ModelDataHistory/{modelName}.csv",
    np.append(yInputData[len(yInputData)-1], [100, 0]).reshape(1, -1),  # Append and reshape to a single row
    delimiter=",",
    fmt="%d",
    header=",".join(list(dataHeader) + ["Cash", "Shares"]),  # Convert header to a string and append new columns
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
                self.add_model()
            elif choice == '3':
                self.remove_model()
            elif choice == '4':
                self.run_simulation()
            elif choice == '5':
                self.check_results()
            elif choice == '6':
                self.check_results()
            elif choice == '7':
                self.check_results()
            elif choice == '8':
                break
            else:
                print("Invalid choice, please try again.")

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
