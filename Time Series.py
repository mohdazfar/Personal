import pandas as pd
import matplotlib as mp
import scipy
import numpy as np
from pandas import Series, DataFrame
import statsmodels



### Check Version ###
print(pd.__version__)
print(mp.__version__)
print(scipy.__version__)


### Date ranges and indexing arrays using series ###
dates = pd.date_range('2012-07-16','2012-07-26')
temp = [43,545,63,432,534,453,23,876,53,53,3]
res = Series(temp,index=dates)
res.values


### Basic DataFrames ###
dates = pd.date_range('2012-07-16','2012-07-26')
temp1 = [43,32,63,98,65,78,23,35,78,56,45]
temp2 = [45,56,6,63,45,64,34,76,34,14,54]
res1 = DataFrame({'NY':temp1,'SA':temp2})
res1['diff'] = res1['NY'] - res1['SA']
# print(res1)
# print(res1.index) #index
# print(res1.values)  #values related to that index
# print(res1.columns) # Columns in the data frame
# print(res1.values[3]) # Particular index values
# print(res1.ix[[0,1,7], ['NY','SA']]) # Returns sliced data frame of having values for 0, 1, 7  and columns NY and SA


### Basic stats functions ###
print(res1.mean(0)) #mean on columns
print(res1.mean(1)) #mean on rows
r = DataFrame([43,32,63,98,65,78,23,35,78,56,45,45,56,6,63,45,64,34,76,34,14,54])
df = pd.rolling_mean(r,5) #Moving averages
print(df)


### Slicing ###
print(res1[res1.NY > 50]) #return the rows with temp greater than 50

### Parse Dates ###
df = pd.read_csv('',parse_dates = ['dates_column'], index_col = 'dates_column') #Give the location and then parse dates for a particular column having dates
# date column will now become the index for the whole data frame

df1 = pd.read_csv('//location', parse_dates = [['date', 'time']]) # combines the dates and time columns if they are not combined



r = statsmodels.tsa.arima_model.ARIMA([43,32,63,98,65,78,23,35,78,56,45,45,56,6,63,45,64,34,76,34,14,54],(1,0,2))
dates = pd.date_range('2012-07-09','2012-07-30')
series = [43.,32.,63.,98.,65.,78.,23.,35.,78.,56.,45.,45.,56.,6.,63.,45.,64.,34.,76.,34.,14.,54.]
res = Series(series, index=dates)
r = ARIMA(res,(1,2,0)).predict()
# resid = r.resid
pred = r.predict('1', '2', typ='levels')
# print(resid)
print(r)

