import requests as r
import pandas as pd

def download(url,filename):
    response = r.get(url)
    if r.status_codes == 200:
        with open(filename,'wb') as f:
            f.write(response.content)

#download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv', 'laptop.csv')
data_to_analyze= pd.read_csv('laptop.csv')
data_to_analyze.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace =True)