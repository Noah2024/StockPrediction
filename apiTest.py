from api import * 

# confirmApiCall("finnhub")
# isMarketOpen()
smaData = getSMA("IBM", timespan="day", adjusted="true", window="50", series_type="close", order="desc")

print(smaData)
print(type(smaData))