from fileinput import filename

import requests as r
import pandas as pd
import matplotlib.pyplot  as py
import seaborn as sn
import numpy as np

def download (filename):
    response = r.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv')
    if (response.status_code == 200):
        with open(filename,'wb') as f:
            f.write(response.content)

file_write = 'laptop.csv'
download(file_write)
laptop_data= pd.read_csv(file_write)
print(laptop_data.head(5).reset_index())

#reg plot
#sn.regplot(x='CPU_frequency', y='Price', data= laptop_data)
#py.show()
print(laptop_data['CPU_frequency'].corr(laptop_data['Price']))
#sn.regplot(x='Screen_Size_inch', y='Price', data= laptop_data)
#py.ylim(0,)
#py.show()
print(laptop_data['Screen_Size_inch'].corr(laptop_data['Price']))

#sn.regplot(x='Weight_pounds', y='Price', data= laptop_data)
#py.ylim(0,)
#py.show()
print(laptop_data['Weight_pounds'].corr(laptop_data['Price']))

#categorical data
sn.boxplot(x='Category', y='Price', data=laptop_data)
py.show()
sn.boxplot(x='OS', y='Price', data=laptop_data)
py.show()

#GROUPBY PIVOT TABLE
data_sub= laptop_data[['GPU','CPU_core','Price']]
data_sub_group = data_sub.groupby(['GPU','CPU_core'], as_index=False).mean()
print(data_sub_group)
data_sub_group_pivot = data_sub_group.pivot(index='GPU',columns='CPU_core')
print(data_sub_group_pivot)

fig, ax = py.subplots()
im = ax.pcolor(data_sub_group_pivot , cmap='RdBu')

#label names
row_labels = data_sub_group_pivot .columns.levels[1]
col_labels = data_sub_group_pivot .index

#move ticks and labels to the center
ax.set_xticks(np.arange(data_sub_group_pivot .shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data_sub_group_pivot .shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)
py.show()




