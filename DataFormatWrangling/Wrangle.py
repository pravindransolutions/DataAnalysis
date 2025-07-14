from json.decoder import NaN

import numpy as np
import pandas as pd
import numpy as n
import requests as req
import matplotlib as plt
from matplotlib import pyplot
from scipy.constants import horsepower
from scipy.ndimage import histogram


# read data
def download( filename, url):
    response = req.get(url, stream = True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(8192):
             f.write(chunk)
             print(chunk)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
filename = 'car.csv'
download(filename, url)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

with open(filename, 'rb') as f:
    chunk_size = 1000
    all_chunks = []
    for read_chunk in pd.read_csv(filename, chunksize= chunk_size):
        all_chunks.append(read_chunk)

all_data = pd.concat(all_chunks)
all_data.columns = headers
all_data.replace('?',n.nan , inplace = True)
missing_data = all_data.isnull()
print(missing_data)
for miss in missing_data.columns.values.tolist():
    print(f'{missing_data[miss].value_counts()} \n')

# fill null data, with means or max value as context until there are no more nan
all_data['normalized-losses'] = all_data['normalized-losses'].astype("float")

nl_mean = all_data['normalized-losses'].mean(axis=0)
#print(nl_mean)
all_data['normalized-losses']= all_data['normalized-losses'].replace(n.nan, nl_mean)

all_data['num-of-doors']= all_data['num-of-doors'].astype("str")
#print(all_data['num-of-doors'].dtype)
#print(all_data['num-of-doors'].value_counts().idxmax())
all_data['num-of-doors'] = all_data['num-of-doors'].replace(n.nan,"four")

all_data['bore'] = all_data['bore'].astype('float')
bore_mean = all_data['bore'].mean(axis=0)
all_data['bore'] = all_data['bore'].replace(n.nan, bore_mean)

rpm_mean=all_data['peak-rpm'].astype('float').mean(axis=0)
#print("Average peak rpm:", rpm_mean)
all_data['peak-rpm'] = all_data['peak-rpm'].replace(n.nan, rpm_mean)

all_data.dropna(subset=["price"], axis=0, inplace=True)
all_data.dropna(subset=["horsepower"], axis=0, inplace=True)

all_data['stroke'] = all_data['stroke'].astype('float')
stroke_mean = all_data['stroke'].mean(axis=0)
all_data['stroke'] = all_data['stroke'].replace(n.nan,stroke_mean)
print(all_data['stroke'])

#set right formats
all_data.reset_index(drop=True, inplace= True)
all_data[['peak-rpm','price']]= all_data[['peak-rpm','price']].astype('float')

#data standardization, convert data to local formats like mpg to lm per litre
all_data['city Litres/100km'] = 235/all_data['city-mpg']
all_data.rename(columns={'city-mpg':'highway-mpg1','city Litres/100km': 'highway Litres/100km'}, inplace=True)
print(all_data.columns)
print(all_data.head())
#data normalization
all_data['length']=all_data['length']/all_data['length'].max()
all_data['width']= all_data['width']/all_data['width'].max()
all_data['height']= all_data['height']/all_data['height'].max()

#Binning horsepower
group_names=['low','medium','high']
all_data['horsepower'] = all_data['horsepower'].astype('int')
bins = np.linspace(all_data['horsepower'].min(axis=0), all_data['horsepower'].max(axis=0),4)
print(bins)

all_data['horsepower-binned']= pd.cut(all_data['horsepower'], bins,labels=group_names , include_lowest=True)
print(all_data['horsepower-binned'])
pyplot.bar(group_names,all_data['horsepower-binned'].value_counts())
pyplot.xlabel('horsepower')
pyplot.ylabel('count')
#pyplot.show()

#indicator variable
indicator_variables = pd.get_dummies(all_data['aspiration'])
all_data = pd.concat([all_data, indicator_variables], axis=1)

# drop original column "aspiration" from "all_data"
all_data.drop("aspiration", axis = 1, inplace=True)
#print(all_data.head())
all_data.to_csv("cleaned_cars.csv")