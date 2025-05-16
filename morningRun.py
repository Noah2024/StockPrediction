import glob
import csv
import time

def makeTrade(tradeData):
    # Placeholder for using API to make trade
    return True

def Main():
    print("test")
    skippedHeader = None
    executedTrades = []
    # Load and execute the trades qued from last night
    with open(f"./Data/tradeQue.csv", "r") as file:
        print("test2")
        reader = csv.reader(file)
        skippedHeader = next(reader)
        print(skippedHeader)
        print("test3")
        print(reader)
        for row in reader:
            print("ROW", row)
            if makeTrade(row):
                # print(f"Trade executed: {row[0]} {row[1]} {row[2]} {row[3]} at {row[4]} on {row[5]}")
                row.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                executedTrades.append(row)
            else:
                print(f"Trade failed: {row[0]} {row[1]} {row[2]} {row[3]} at {row[4]} on {row[5]}")

    #Add to tradeLog.csv
    with open(f"tradeLog.csv", "a", newline="") as file:
        writer = csv.writer(file)
        for trade in executedTrades:
            writer.writerow(trade)

    #Clear the tradeQue.csv
    with open(f"./Data/tradeQue.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(skippedHeader)    
        
    # Placeholder for using API to make trade

if __name__ == "__main__":
    Main()