import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('cleaned_cars.csv')
print(df['normalized-losses'].dtypes)
print(df[['normalized-losses','price']].corr(numeric_only=False))

#correlation for a subset
print(df[['bore','stroke','compression-ratio','horsepower']].corr())

sn.regplot(x='engine-size', y='price', data=df)
#plt.show()

#corerelation between price and stroke
print(df['stroke'].corr(df['price'])) #0.0947 weak
sn.regplot(x='stroke', y='price', data=df)
#plt.show()

#categorical variables, box plot
sn.boxplot(x='body-style', y='price', data= df)
#plt.show()

sn.boxplot(x='engine-location', y='price', data=df)
#plt.show()

print(df.describe(include=['object']))
# value count on categorical data

df_count=df[['drive-wheels']].value_counts().to_frame()
df_count= df_count.reset_index()
df_count=df_count.rename(columns={'drive-wheels': 'value_count'})
print(df_count)

#grouped
df_grouped_set = df[['drive-wheels','body-style','price']]
df_grouped = df_grouped_set.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(df_grouped)
grouped_pivot=df_grouped.pivot(index="drive-wheels", columns="body-style", values="price")
print(grouped_pivot)

#another example with body-styles
df_grouped_styles_set = df[['body-style','price']]
df_grouped_styles = df_grouped_styles_set.groupby(['body-style'],as_index=False).mean()
df_pivot = df_grouped_styles.pivot(index='body-style', columns='price')
print(df_pivot)
plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()