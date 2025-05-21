
#This will be the script to start the program and run the main menu
#Create New Models, Add them, remove them, run simulations, and check results
import traceback
import importlib
class MainStart:
    
    def __init__(self):
        self.models = []  # List to store models
        self.current_model = None  # Currently selected model

    def mainMenu(self):
        while True:
            import util
            importlib.reload(util)
            from util import getJsonValue
            simStatus  = getJsonValue("./Data/systemMetaData.json", "status")
            print("-----------------------")
            print("Main Menu:")
            print(f"Simulation Status: {simStatus}")
            print("-----------------------")
            print("1. Create New Model")
            print("2. Add Model To Simulation")
            print("3. Archive Model")
            print("4. Start Simulation")
            print("5. End Simulation")
            print("6. Check Current Results")
            print("7. Single Predict Model")
            print("8. Exit")

            choice = input("Enter your choice: ")

            if choice == '1':
                self.createNewModel()
            elif choice == '2':
                self.addFromArchive()
            elif choice == '3':
                self.archiveModel()
            elif choice == '4':
                self.startSimulation()
            elif choice == '5':
                self.stopSimulation()
            elif choice == '6':
                self.checkResults()
            elif choice == "7":
                self.singlePredict()
            elif choice == '8':
                break
            else:
                print("Invalid choice, please try again.")

    def singlePredict(Self):
        try:
            import tensorFlowModel
            importlib.reload(tensorFlowModel)
            from tensorFlowModel import singlePredict
            singlePredict()
        except Exception as ex:
            print("Exception in single predict")
            traceback.print_exc()

    def startSimulation(Self):
        try:
            import crono
            importlib.reload(crono)
            from crono import startSimulation
            
            startSimulation()
        except Exception as ex:
            print("Exception in starting simulation")
            traceback.print_exc()
        
    
    def stopSimulation(Self):
        try:
            import crono
            importlib.reload(crono)
            from crono import stopSimulation
            stopSimulation()
        except Exception as ex:
            print("Exception in stopping simulation")
            traceback.print_exc()
        
    
    def addFromArchive(Self):
        try:
            import archive
            importlib.reload(archive)
            from archive import addFromArchive
            addFromArchive()
        except Exception as ex:
            print("Exception in adding from archive")
            traceback.print_exc()
        

    def archiveModel(Self):#Disabled for testing purposes
        try:
            import archive
            importlib.reload(archive)
            from archive import archiveModel
            archiveModel()
        except Exception as ex:
            print("Exception in archiving model")
            traceback.print_exc()
    

    def createNewModel(Self):
        from util import yesNoInput
        from util import selectFile
        import tensorFlowModel
        importlib.reload(tensorFlowModel)
        from tensorFlowModel import tensorFlowModel 

        modelName = input("Enter the name of the new model: ")
        default = yesNoInput("Do you want to load the default model template? (y/n): ")
        try: 
            from tensorFlowModel import tensorFlowModel
            if default:
                print("Loading default model template...")
                model, config = tensorFlowModel("ModelTemplates/default.json", modelName)
                #breakpoint()
            else:
                print("Enter the path to an alternative .json model template:")
                selectFile()
            return
        except Exception as ex:
            print("Exception in starting simulation")
            traceback.print_exc()
    
    def checkResults(Self):
        try:
            import analysis
            importlib.reload(analysis)
            from analysis import checkResults 
            checkResults()
        except Exception as ex:
            print("Exception in checking results")
            traceback.print_exc() 
        
        
        

MainMenu = MainStart()
MainMenu.mainMenu()
