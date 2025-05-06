import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

def Main():
    if len(sys.argv) < 2:
        print("Usage: py CreateModel.py <templatePath>")

    templatePath = sys.argv[1]
    print("Template Path:", templatePath)
    config = None
    with open(templatePath, 'r') as file:
        config = json.load(file)
        print("PATH:", config["xInputPath"])
        xInputData = np.loadtxt(f'{config["xInputPath"]}', delimiter=',', skiprows=1)#
        yInputData = np.loadtxt(f'{config["yInputPath"]}', delimiter=',', skiprows=1)

    finalActiveFunc = None if config["modelType"] == "regression" else 'softmax'

    model = keras.Sequential()
    print("SHAPE", xInputData.shape)
    model.add(keras.layers.InputLayer(shape=xInputData[0].shape))  # Input layer expects data with shape `input_shape`
    for i in range(len(config["architecture"])-1):
        model.add(keras.layers.Dense(config["architecture"][i], activation=config["activation"]))
    model.add(keras.layers.Dense(config["numClasses"], activation=finalActiveFunc))
    model.compile(optimizer=config["optimizer"], loss=config["lossFunction"], metrics=config["metrics"])
    
    dataset = tf.data.Dataset.from_tensor_slices((xInputData, yInputData))
    dataset = dataset.batch(config["batchSize"]).prefetch(tf.data.AUTOTUNE)

    # model.fit(dataset, verbose = 0)  # Example training call
    
    finalPath = config["modelPath"] + config["modelName"] + ".keras"
    print("Model Created")
    print("Fitting Model...")
    history = model.fit(dataset, epochs=config["epochs"],batch_size=config["batchSize"], verbose=0)
    # print("Model Fitted")
    # print(history.history)

    pred = model.predict(xInputData[0][np.newaxis, :])
    print("Predicted Value:", pred)
    print("Actual Value:", yInputData[0])
    print("Diff Between Predicted and Actual:", pred[0] - yInputData[0])
    model.save(finalPath)  # Save the model to the specified path
    
    # model.predict(xInputData[len(xInputData)-1])  # Example prediction call
    # Extracting the required fields from the loaded data
if __name__ == "__main__":
    Main()