import requests as req
import numpy as n
import sklearn as sl
from sklearn.linear_model import   LinearRegression
from sklearn.ridge import Rigde
from sklearn.metrics import mean_squared_error as mse, mean_squared_error
import pandas as pd
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plot


def download(filename, url):
    response = req.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)
            print(chunk)


url_ref = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
file_name = 'car.csv'
download(file_name, url_ref)
lm= LinearRegression()

data_to_analyse = pd.read_csv('car.csv')
data_to_analyse.columns=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
#clean the data
#fill null with mean or 0 or drop them as necessary
print(data_to_analyse.columns)
data_to_analyse.replace('?',n.nan, inplace= True)
missing_data = data_to_analyse.isnull()
for miss in missing_data.columns.values.tolist():
    print(f'{missing_data[miss].value_counts()}\n')

#drop data with price =?
#horsepower has 2 nulled values, dropping those rows

data_to_analyse.dropna(subset='price', axis=0, inplace=True)
data_to_analyse.dropna(subset='horsepower', axis=0, inplace= True)
data_to_analyse['normalized-losses'] = data_to_analyse['normalized-losses'].astype('float')
nl_mean= data_to_analyse['normalized-losses'].mean(axis=0)
data_to_analyse.reset_index(inplace=True)
data_to_analyse['price']=data_to_analyse['price'].astype('float')
data_to_analyse['normalized-losses'] = data_to_analyse['normalized-losses'].replace(n.nan, nl_mean)
data_subset= data_to_analyse[['highway-mpg','price']]

#LINEAR REGRESSION
x=data_to_analyse[['highway-mpg']]
y=data_to_analyse['price']

print(y.dtypes)
lm.fit(x,y)
print(lm.predict(x))

print(lm.coef_)
print(lm.intercept_)
yhat=lm.intercept_-(lm.coef_*30)
#print(yhat)

width = 12
height = 10
plot.figure(figsize=(width, height))
sb.regplot(x='highway-mpg', y ='price', data=data_to_analyse)
plot.ylim(0,)
plot.show()
#quantative calculations to determine if the model is a good fit
#R2 AND MSE(MEAN SCORE ERROR)
print(lm.score(x,y))
yhat=lm.predict(x)
mse_slr=mean_squared_error(data_to_analyse['price'],yhat)
print(f'mean_slr={mse_slr}')

# see the difference between the redicted values and real values
plot.plot(data_to_analyse['price'],yhat)
plot.show()


#training the model using engine size and price
lm1 = LinearRegression()
x1= data_to_analyse[['engine-size']]

y1=data_to_analyse['price']
lm1.fit(x1,y1)
engine_size = 40
#yhat=lm1.intercept_+(lm1.coef_*engine_size)
print(f'{lm1.intercept_}+{lm1.coef_}*engine_size')
sb.regplot(x='engine-size', y='price', data=data_to_analyse)
plot.show()

#check if prediction and real values differ
new_input=n.arange(1, 100, 1).reshape(-1, 1)
plot.plot(new_input,lm1.predict(new_input))
plot.show()

#multiple linear regression: indepedent variables horsepower,engine-size,curb-weight and hignway-mpg
lm2 = LinearRegression()
x2 = data_to_analyse[['horsepower','engine-size','curb-weight','highway-mpg']]
y2 = data_to_analyse[['price']]
lm2.fit(x2,y2)
co_eff=lm2.coef_
print(lm2.coef_)
#yhat
print(f'{lm2.intercept_}+{co_eff[0,0]}*horsepower+{co_eff[0,1]*engine_size}+{co_eff[0,2]}*curb-weight+{co_eff[0,3]}*highway-mpg')
#visualize multiple linear regression with ditribution plot
#yhat2=lm2.predict(x2)
#ax1= sb.displot(data_to_analyse['price'], kde=True)
#sb.displot(yhat2, ax=ax1)
#plot.title('Actual vs Fitted Values for Price')
#plot.xlabel('Price (in dollars)')
#plot.ylabel('Proportion of Cars')

#quantitive analysis for the fit
print(f'score for mlr={lm2.score(x2,y2)}') #.81, 81 percent of the time the price is predicted correctly
yhat=lm2.predict(x2)
mse_mlr=mean_squared_error(y2,yhat)
print(f'mse_mlr={mse_mlr}')
#plot.show()
#Create and train a Multiple Linear Regression model "lm2" where the response variable is "price", and the predictor variable is "normalized-losses" and "highway-mpg".
lm3 = LinearRegression()
x3= data_to_analyse[['normalized-losses','highway-mpg']]
y3=data_to_analyse['price']
lm3.fit(x3,y3)
print(lm3.coef_)

#compare correlation between highway-mpg and engine-size with price
corr_high = data_to_analyse[['highway-mpg','price']].corr()
corr_size=data_to_analyse[['engine-size','price']].corr()
print(f'corr_high={corr_high}, corr_size={corr_size}')

#residual plot, the points are not randomly spaced around the horizontal line linear regression is not the right way to modle
sb.residplot(x=data_to_analyse['highway-mpg'],y=data_to_analyse['price'])
plot.show()

#validate model by using R2 and mse, R2 needs to be high mse needs to ne low between the models
#polynomical regression is the best fit( not here though)
