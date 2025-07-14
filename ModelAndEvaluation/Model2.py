import requests as r
import pandas as pd
import matplotlib.pyplot  as py
import seaborn as sn
import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def download (filename):
    response = r.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv')
    if (response.status_code == 200):
        with open(filename,'wb') as f:
            f.write(response.content)

file_write = 'laptop.csv'
download(file_write)
data_to_analyse = pd.read_csv('laptop.csv')
print(data_to_analyse.head(5))
#SLR
lm1= LinearRegression()
x1= data_to_analyse[['CPU_frequency']]
y1=data_to_analyse['Price']
lm1.fit(x1,y1)
yhat=lm1.predict(x1)
print(f'yhat={lm1.intercept_}+{lm1.coef_}*price')
R2=lm1.score(x1,y1)
print(R2)
mse= mean_squared_error(data_to_analyse['Price'],yhat)
print(mse)
#distplot

#multiple Linear regression
lm2=LinearRegression()
x2=data_to_analyse[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
y2=data_to_analyse['Price']
lm2.fit(x2,y2)
yhat2=lm2.predict(x2)
r22=lm2.score(x2,y2)
print(r22)
mse2=mean_squared_error(data_to_analyse['Price'],yhat2)
print(mse)
#distplot

#polynomial regression
x3= data_to_analyse['CPU_frequency']
y3= data_to_analyse['Price']

f1= np.polyfit(x3,y3,1)
p1=np.poly1d(f1)

f2=np.polyfit(x3,y3,3)
p2=np.poly1d(f2)

f3=np.polyfit(x3,y3,5)
p3=np.poly1d(f3)

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    py.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    py.title(f'Polynomial Fit for Price ~ {Name}')
    ax = py.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = py.gcf()
    py.xlabel(Name)
    py.ylabel('Price of laptops')
    py.show()

PlotPolly(p1,x3,y3,"CPU_FREQUENCY")
PlotPolly(p2,x3,y3,"CPU_FREQUENCY")
PlotPolly(p3,x3,y3,"CPU_FREQUENCY")
#R2
print(f'r2 for linear regression {r2_score(y3,p1(x3))}')
print(f'r2 for a three degree polynomial {r2_score(y3,p2(x3))}')
print(f'r2 for a five degree polynomial {r2_score(y3,p3(x3))}')

print(f'mse for linear relationship {mean_squared_error(y3,p1(x3))}')
print(f'mse for linear relationship {mean_squared_error(y3,p2(x3))}')
print(f'mse for linear relationship {mean_squared_error(y3,p3(x3))}')





