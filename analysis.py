import os 
import pandas as pd
import matplotlib.pyplot as plt

def checkResults():
        for file in os.listdir("./ModelDataHistory"):
            if file.endswith(".csv"): 
                print(f"Processing file: {file}")
                df = pd.read_csv("./ModelDataHistory/" + file)
                # Plot specific columns
                equity = (df["close"] * df["Shares"])
                print("---")
                print(df["Cash"])
                print("---")
                print(equity)
                plt.plot(df["timestamp"], equity, label="Open vs High")
                plt.xlabel("Index")
                plt.ylabel("Close Price")
                plt.title(f"Preformance of Model {file}")
                plt.legend()
                plt.show()