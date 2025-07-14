import requests as r
import pandas as p
import numpy as np
import seaborn as sn
import matplotlib as m
import matplotlib.pyplot as py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def download(filename):
    response = r.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv')
    if response.status_code == 200:
        with open(filename,'wb') as f:
            f.write(response.content)

filename= "insurance.csv"
#download(filename)
insurance_data_to_analyze= p.read_csv(filename)
insurance_data_to_analyze.columns = ['Age','Gender','BMI','Number_of_children','Smoker','Region','Charges']
print(insurance_data_to_analyze.head(5))

insurance_data_to_analyze.replace('?',np.nan, inplace=True)
missing_data=insurance_data_to_analyze.isnull()
for miss in missing_data.columns.values.tolist():
    print(f'{missing_data[miss].value_counts()} \n')

print(insurance_data_to_analyze.info())

#replace age nan with mean, smoker category with max valuecount in the column
mean_age= insurance_data_to_analyze['Age'].astype('float').mean()
insurance_data_to_analyze['Age']= insurance_data_to_analyze['Age'].replace(np.nan,mean_age)
mode=insurance_data_to_analyze['Smoker'].value_counts().idxmax()
insurance_data_to_analyze['Smoker'].replace(np.nan,mode,inplace=True)
print(insurance_data_to_analyze.info())
insurance_data_to_analyze[['Age','Smoker']] = insurance_data_to_analyze[['Age','Smoker']].astype(int)

# round charges
print(insurance_data_to_analyze['Charges'].head(5))
insurance_data_to_analyze['Charges'] = insurance_data_to_analyze['Charges'].round(2)
print(insurance_data_to_analyze['Charges'])

#
sn.regplot(x = 'Charges', y ='BMI', data=insurance_data_to_analyze)
py.show()

sn.boxplot(x='Smoker',y='Charges',data=insurance_data_to_analyze)
py.show()

print(insurance_data_to_analyze.corr())
#Linear regression
lr = LinearRegression()
lr.fit(insurance_data_to_analyze[['Smoker']],insurance_data_to_analyze[['Charges']])
print(lr.score(insurance_data_to_analyze[['Smoker']],insurance_data_to_analyze[['Charges']]))

x=insurance_data_to_analyze[['Age','Gender','BMI','Number_of_children','Smoker','Region']]
y=insurance_data_to_analyze['Charges']
xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size=0.2, random_state=0)

#ridge regression
rr= Ridge(alpha=0.1)
rr.fit(xtrain,ytrain)
yhat=rr.predict(xtrain)
rr.score(ytest,yhat)

pr=PolynomialFeatures(2)
xtrainpr= pr.fit_transform(xtrain,ytrain)
xtestpr=pr.fit_transform(xtestpr, ytest)
rr.fit(xtrainpr,ytrain)
yhat=rr.predict(xtestpr)
rr.score(ytest,yhat)