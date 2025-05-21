import os 

def addFromArchive():
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

def archiveModel():#Disabled for testing purposes
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