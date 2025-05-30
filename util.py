import json
import os
import platform

def is_windows():#Chatgpt
    return platform.system().lower().startswith("win")

def setJsonValue(path, key, value):#ChatGpt
    with open(path, "r") as f:
        data = json.load(f)
    data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def getJsonValue(path, key=None, default=None):#ChatGpt
    try:
        with open(path, "r") as f:
            data = json.load(f)
            # print("DATA AS", data)
        return data[key] if key else data
    except (FileNotFoundError, json.JSONDecodeError):
        return default if key else {}
    except KeyError:
        return default

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


# def loadTemplateData(templatePath):
#     dataHeader = None

#     with open(templatePath, 'r') as file:
#         data = json.load(file)
#         print("PATH:", data["xInputPath"])
#         xInputData = np.loadtxt(f'{data["xInputPath"]}', delimiter=',', skiprows=1)#
#         yInputData = np.loadtxt(f'{data["yInputPath]"]}', delimiter=',', skiprows=1)
#         dataHeader = np.loadtxt(f'{data["xInputPath"]}', delimiter=',', max_rows=1, dtype=str)  
#     # Extracting the required fields from the loaded data
#     return data, xInputData, yInputData