import json
import numpy as np

def Main():
    config = None
    with open(templatePath, 'r') as file:
        data = json.load(file)
        print("PATH:", data["xInputPath"])
        xInputData = np.loadtxt(f'{data["xInputPath"]}', delimiter=',', skiprows=1)#
        yInputData = np.loadtxt(f'{data["yInputPath]"]}', delimiter=',', skiprows=1)

    finalActiveFunc = None if config.modelType == "regression" else 'softmax'

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=xInputData.shape))  # Input layer expects data with shape `input_shape`
    for i in range(len(config["architecture"])-1):
        model.add(keras.layers.Dense(config["architecture"][i], activation=config["activationFunc"]))
    model.add(keras.layers.Dense(config.numClasses, activation=finalActiveFunc))
    model.compile(optimizer=config["optimizer"], loss=config["lossFunction"], metrics=["metrics"])
    
    # Extracting the required fields from the loaded data
if __name__ == "__main__":
    Main()