import statistics

from numpy.polynomial.polynomial import Polynomial
from scipy.cluster.hierarchy import optimal_leaf_ordering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


import requests as r
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def download ():
    response = r.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv','module_5_auto.csv')
    if (response.status_code == 200):
        with open("mod.csv",'wb') as f:
            f.write(response.content)

download()
data_to_evaluate = pd.read_csv('mod.csv')
print(data_to_evaluate.head())
print(data_to_evaluate.columns)
data_to_evaluate.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1 , inplace=True)
#read numerc data
data_to_evaluate = data_to_evaluate._get_numeric_data()
print(data_to_evaluate.head())
print(data_to_evaluate.columns)
print(data_to_evaluate.isnull())


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    """
    DITRIBUTION PLOT for ytrain and ytest
    :param RedFunction:
    :param BlueFunction:
    :param RedName:
    :param BlueName:
    :param Title:
    """
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#cleandata
missing_data= data_to_evaluate.isnull()
for miss in missing_data.columns.values.tolist():
    print(f'{missing_data[miss].value_counts()} \n')

#drop all nan
#split data set into train and test
data_to_evaluate.dropna(inplace=True)
y= data_to_evaluate['price']
x=data_to_evaluate.drop('price',axis=1)
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.4)
print(x_test)

lr = LinearRegression()

yhat = lr.fit(x_train[['horsepower']], y_train)
score_train= lr.score(x_train[['horsepower']],y_train)
score_test=lr.score(x_test[['horsepower']],y_test)
print(f'score from trianing data={score_train}, score from test data={score_test}')

#sometimes we might not have enough test data then we need to do cross validation and determine the
#score
rscore=cross_val_score(lr,x[['horsepower']], y, cv=4) #cv = number of folds or subset for splitting data 3 for training and
#1 for testing. for subset1 as trining, subset2 as training, subset3 as training and sunset4 as test determine score
#for subset1,3and4 as training and 2 as test finds score, for various subset as test find score. Average it and
# is returned
print(rscore)
print(f'score. with 4 folds = {rscore.mean()}')

#cross validation with 2 folds
rscore2=cross_val_score(lr,x[['horsepower']],y, cv=2)

#print the rscore.mean
print(f'rscore for 2 fold = {rscore2.mean()}')


#the model trained by training data and when we apply th etest data
#the model may not respons well to the test data sometimes because of overfitting
#linear regression with trained and test data and compare
lr3= LinearRegression()
lr3.fit(x_train[['horsepower','curb-weight', 'engine-size','highway-mpg']], y_train)
yhat_train=lr3.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
yhat_test=lr3.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])

#distribution plot train prediction vs test prediction, in this case the train predictions are better
#DistributionPlot(y_train, yhat_train,'Actual values', 'Predicted Values', 'Actual vs Predicted')
#DistributionPlot(y_test, yhat_test,'Test values', 'Predicted Values', 'Test vs Predicted')

#overfitting happens when the model trains with the noise rather than the underlying process itself
#explaining with polynomial regression
#with 55 percent test data
x_train,x_test,y_train,y_test= train_test_split(x[['horsepower']],y,test_size=0.45)

pr = PolynomialFeatures(5)
pr_train = pr.fit_transform(x_train)
pr_test =pr.fit_transform(x_test)
#print(pr_train)

poly = LinearRegression()
poly.fit(pr_train,y_train)
yhat_poly= poly.predict(pr_test)
#plot predicted vs test values
PollyPlot(x_train, x_test,y_train,y_test, poly, pr)
print(poly.score(pr_train, y_train))
print(poly.score(pr_test,y_test))

#plot mse against order of polynomial
mse=[]
order=[1,2,3,4]
for i in order:
    pr= PolynomialFeatures(i)
    pr_train=pr.fit_transform(x_train)
    pr_test=pr.fit_transform(x_test)
    poly.fit(pr_train, y_train)
    mse.append(poly.score(pr_test,y_test))

plt.plot(order,mse)
plt.title('mse vs order')
plt.xlabel('order')
plt.ylabel('mse')
plt.show()

#ridge regression, when there is a corellation between columns used for modeling
#test data is used as validation data
#alpha is a hyper parameter used to scale the coefficients of the polynomial
x_train,x_test,y_train,y_test= train_test_split(x[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']],y,test_size=0.45)
pr = PolynomialFeatures(2)
pr_train= pr.fit_transform(x_train)
pr_test=pr.fit_transform(x_test)
rr = Ridge(1)
rr.fit(pr_train,y_train)

yhat_rr= rr.predict(pr_test)
print(f'predicted yhat with ridge= {yhat_rr[0:5]}')
print(f'actual values:{y_test[0:5].values}')

#finding the best alpha value, as the alphaincreases, the scores would meet at a point
alpha=[0.001,0.01,.01,1, 100, 1000, 1000000]
tests=[]
trains=[]

for i in alpha:
    rr= Ridge(i)
    rr.fit(pr_train, y_train)
    test_score,train_score= rr.score(pr_test,y_test), rr.score(pr_train, y_train)
    tests.append(test_score)
    trains.append(train_score)

width = 12
height = 10
plt.figure(figsize=(width, height))
plt.plot(alpha, tests, label= 'test score')
plt.plot(alpha, trains, label = 'train score')
plt.xlabel('alpha')
plt.ylabel('score')
plt.legend()
plt.show()
