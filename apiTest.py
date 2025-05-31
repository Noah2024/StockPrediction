from api import * 

data = getAllHistoric("AAPL")
print(isMarketOpen())
print(data.head())
print(type(data))
# url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=H3S85LY8M5OL60UU&datatype=csv&outputsize=compact"
# r = requests.get(url)
# # print(r.text[:500])
# data = pd.read_csv(StringIO(r.text), index_col=None)
# print(data.head())