The Idea:

So pretty basic idea, I wanted to predict the stock market, "wow no one has ever thought of doing that", I hear you saying.
I am well aware that this task is neigh impossible even for those much smarter than I, but idrc, its fun to try. To start
I just wanted to imput very basic data (start, high, low, close) as the inputs into a NNW, and have that output what the 
next days predictions might look like, that basic idea spirled out into a creating an API managment and Cache system, 
along with the facilites to track (quite poorly, but track nonthe less) the progress of different models. 


General ImplementationNotes:

Morning and Evening Run are scedhuled to run 5 minutes after and 5 minutes before the open and end of markets respectivley
This is becuase they should only run if the markets were open that day, and I can only get data from every 5 minute interval

In the systemMetaData the limit for API calls are represented as "alphaVanLimit": "25 - 1440", the first number being the
Total number of API calls that can be made, and the number after is how many MINUTES it takes for that number to reset

