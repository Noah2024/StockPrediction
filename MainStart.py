#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
import tensorflow as tf
from tensorflow import keras
import os

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
            
def tensorFlowModel(inputShape, numClasses, architecture):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))  # Input layer expects data with shape `input_shape`
        for i in range(len(architecture)-1):
            model.add(keras.layers.Dense(architecture[i], activation='relu'))
        model.add(keras.layers.Dense(numClasses, activation=None))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model
class MainStart:
    def __init__(self):
        self.models = []  # List to store models
        self.current_model = None  # Currently selected model

    def mainMenu(self):
        while True:
            print("-----------------------")
            print("Main Menu:")
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
        advParam = yesNoInput("Do you want to load an advanced template? (y/n): ")
        if advParam:
            print("Advanced parameters can be set here.")
            # Placeholder for advanced parameters
        else:
            print("No advanced parameters selected.")
            print("Please Select InputX Data to train on")
            InputDataX = selectFile()
            print("Please Select InputY Data to train on")
            InputDataY = selectFile()
            arch = SelectModelArchitecture()
            print(modelName or InputDataX or InputDataY or arch)
            if modelName == None or InputDataX == None or InputDataY == None or arch == None:
                print("Invalid Paramters: Model creation cancelled.")
                return None
        
        

    
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