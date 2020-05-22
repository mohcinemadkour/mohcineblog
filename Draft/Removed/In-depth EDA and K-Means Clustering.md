title: In-depth EDA and K-Means Clustering
Date: 2019-10-11 13:01
Category: Machine Learning, EDA
Tags: Machine Learning, Clustering, Kmeans
Slug: Machine Learning, Clustering, Kmeans
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com


Our world population is expected to grow from 7.3 billion today to 9.7 billion in the year 2050. Finding solutions for feeding the growing world population has become a hot topic for food and agriculture organizations, entrepreneurs and philanthropists. These solutions range from changing the way we grow our food to changing the way we eat. To make things harder, the world's climate is changing and it is both affecting and affected by the way we grow our food – agriculture. This dataset provides an insight on our worldwide food production - focusing on a comparison between food produced for human consumption and feed produced for animals.

The Food and Agriculture Organization of the United Nations provides free access to food and agriculture data for over 245 countries and territories, from the year 1961 to the most recent update (depends on the dataset). One dataset from the FAO's database is the Food Balance Sheets. It presents a comprehensive picture of the pattern of a country's food supply during a specified reference period, the last time an update was loaded to the FAO database was in 2013. The food balance sheet shows for each food item the sources of supply and its utilization. This chunk of the dataset is focused on two utilizations of each food item available:

Food - refers to the total amount of the food item available as human food during the reference period.
Feed - refers to the quantity of the food item available for feeding to the livestock and poultry during the reference period.
Dataset's attributes:

Area code - Country name abbreviation
Area - County name
Item - Food item
Element - Food or Feed
Latitude - geographic coordinate that specifies the north–south position of a point on the Earth's surface
Longitude - geographic coordinate that specifies the east-west position of a point on the Earth's surface
Production per year - Amount of food item produced in 1000 tonnes

This is a simple exploratory notebook that heavily expolits pandas and seaborn

The dataset and the notebook can be found at this kaggle competition: https://www.kaggle.com/mmadkour/in-depth-eda-and-k-means-clustering


```python
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# importing data
df = pd.read_csv("../input/FAO.csv",  encoding = "ISO-8859-1")
pd.options.mode.chained_assignment = None
```

Let's see what the data looks like...


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Area Abbreviation</th>
      <th>Area Code</th>
      <th>Area</th>
      <th>Item Code</th>
      <th>Item</th>
      <th>Element Code</th>
      <th>Element</th>
      <th>Unit</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>...</th>
      <th>Y2004</th>
      <th>Y2005</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>AFG</td>
      <td>2</td>
      <td>Afghanistan</td>
      <td>2511</td>
      <td>Wheat and products</td>
      <td>5142</td>
      <td>Food</td>
      <td>1000 tonnes</td>
      <td>33.94</td>
      <td>67.71</td>
      <td>...</td>
      <td>3249.0</td>
      <td>3486.0</td>
      <td>3704.0</td>
      <td>4164.0</td>
      <td>4252.0</td>
      <td>4538.0</td>
      <td>4605.0</td>
      <td>4711.0</td>
      <td>4810</td>
      <td>4895</td>
    </tr>
    <tr>
      <td>1</td>
      <td>AFG</td>
      <td>2</td>
      <td>Afghanistan</td>
      <td>2805</td>
      <td>Rice (Milled Equivalent)</td>
      <td>5142</td>
      <td>Food</td>
      <td>1000 tonnes</td>
      <td>33.94</td>
      <td>67.71</td>
      <td>...</td>
      <td>419.0</td>
      <td>445.0</td>
      <td>546.0</td>
      <td>455.0</td>
      <td>490.0</td>
      <td>415.0</td>
      <td>442.0</td>
      <td>476.0</td>
      <td>425</td>
      <td>422</td>
    </tr>
    <tr>
      <td>2</td>
      <td>AFG</td>
      <td>2</td>
      <td>Afghanistan</td>
      <td>2513</td>
      <td>Barley and products</td>
      <td>5521</td>
      <td>Feed</td>
      <td>1000 tonnes</td>
      <td>33.94</td>
      <td>67.71</td>
      <td>...</td>
      <td>58.0</td>
      <td>236.0</td>
      <td>262.0</td>
      <td>263.0</td>
      <td>230.0</td>
      <td>379.0</td>
      <td>315.0</td>
      <td>203.0</td>
      <td>367</td>
      <td>360</td>
    </tr>
    <tr>
      <td>3</td>
      <td>AFG</td>
      <td>2</td>
      <td>Afghanistan</td>
      <td>2513</td>
      <td>Barley and products</td>
      <td>5142</td>
      <td>Food</td>
      <td>1000 tonnes</td>
      <td>33.94</td>
      <td>67.71</td>
      <td>...</td>
      <td>185.0</td>
      <td>43.0</td>
      <td>44.0</td>
      <td>48.0</td>
      <td>62.0</td>
      <td>55.0</td>
      <td>60.0</td>
      <td>72.0</td>
      <td>78</td>
      <td>89</td>
    </tr>
    <tr>
      <td>4</td>
      <td>AFG</td>
      <td>2</td>
      <td>Afghanistan</td>
      <td>2514</td>
      <td>Maize and products</td>
      <td>5521</td>
      <td>Feed</td>
      <td>1000 tonnes</td>
      <td>33.94</td>
      <td>67.71</td>
      <td>...</td>
      <td>120.0</td>
      <td>208.0</td>
      <td>233.0</td>
      <td>249.0</td>
      <td>247.0</td>
      <td>195.0</td>
      <td>178.0</td>
      <td>191.0</td>
      <td>200</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>



# Plot for annual produce of different countries with quantity in y-axis and years in x-axis


```python
area_list = list(df['Area'].unique())
year_list = list(df.iloc[:,10:].columns)

plt.figure(figsize=(24,12))
for ar in area_list:
    yearly_produce = []
    for yr in year_list:
        yearly_produce.append(df[yr][df['Area'] == ar].sum())
    plt.plot(yearly_produce, label=ar)
plt.xticks(np.arange(53), tuple(year_list), rotation=60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0.)
plt.savefig('p.png')
plt.show()
```


![png](/images/output_5_0.png)


Clearly, China, India and US stand out here. So, these are the countries with most food and feed production.

Now, let's have a close look at their food and feed data

# Food and feed plot for the whole dataset


```python
sns.factorplot("Element", data=df, kind="count")
plt.show()
```


![png](/images/output_7_0.png)


So, there is a huge difference in food and feed production. Now, we have obvious assumptions about the following plots after looking at this huge difference.

# Food and feed plot for the largest producers(India, USA, China)


```python
sns.factorplot("Area", data=df[(df['Area'] == "India") | (df['Area'] == "China, mainland") | (df['Area'] == "United States of America")], kind="count", hue="Element", size=8, aspect=.8)
```




    <seaborn.axisgrid.FacetGrid at 0x7fe39e56a5f8>




![png](/images/output_9_1.png)


Though, there is a huge difference between feed and food production, these countries' total production and their ranks depend on feed production.

Now, we create a dataframe with countries as index and their annual produce as columns from 1961 to 2013.


```python
new_df_dict = {}
for ar in area_list:
    yearly_produce = []
    for yr in year_list:
        yearly_produce.append(df[yr][df['Area']==ar].sum())
    new_df_dict[ar] = yearly_produce
new_df = pd.DataFrame(new_df_dict)

new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Afghanistan</th>
      <th>Albania</th>
      <th>Algeria</th>
      <th>Angola</th>
      <th>Antigua and Barbuda</th>
      <th>Argentina</th>
      <th>Armenia</th>
      <th>Australia</th>
      <th>Austria</th>
      <th>Azerbaijan</th>
      <th>...</th>
      <th>United Republic of Tanzania</th>
      <th>United States of America</th>
      <th>Uruguay</th>
      <th>Uzbekistan</th>
      <th>Vanuatu</th>
      <th>Venezuela (Bolivarian Republic of)</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>Zambia</th>
      <th>Zimbabwe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>9481.0</td>
      <td>1706.0</td>
      <td>7488.0</td>
      <td>4834.0</td>
      <td>92.0</td>
      <td>43402.0</td>
      <td>0.0</td>
      <td>25795.0</td>
      <td>22542.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>12367.0</td>
      <td>559347.0</td>
      <td>4631.0</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>9523.0</td>
      <td>23856.0</td>
      <td>2982.0</td>
      <td>2976.0</td>
      <td>3260.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9414.0</td>
      <td>1749.0</td>
      <td>7235.0</td>
      <td>4775.0</td>
      <td>94.0</td>
      <td>40784.0</td>
      <td>0.0</td>
      <td>27618.0</td>
      <td>22627.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>12810.0</td>
      <td>556319.0</td>
      <td>4448.0</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>9369.0</td>
      <td>25220.0</td>
      <td>3038.0</td>
      <td>3057.0</td>
      <td>3503.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>9194.0</td>
      <td>1767.0</td>
      <td>6861.0</td>
      <td>5240.0</td>
      <td>105.0</td>
      <td>40219.0</td>
      <td>0.0</td>
      <td>28902.0</td>
      <td>23637.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>13109.0</td>
      <td>552630.0</td>
      <td>4682.0</td>
      <td>0.0</td>
      <td>103.0</td>
      <td>9788.0</td>
      <td>26053.0</td>
      <td>3147.0</td>
      <td>3069.0</td>
      <td>3479.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10170.0</td>
      <td>1889.0</td>
      <td>7255.0</td>
      <td>5286.0</td>
      <td>95.0</td>
      <td>41638.0</td>
      <td>0.0</td>
      <td>29107.0</td>
      <td>24099.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>12965.0</td>
      <td>555677.0</td>
      <td>4723.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>10539.0</td>
      <td>26377.0</td>
      <td>3224.0</td>
      <td>3121.0</td>
      <td>3738.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10473.0</td>
      <td>1884.0</td>
      <td>7509.0</td>
      <td>5527.0</td>
      <td>84.0</td>
      <td>44936.0</td>
      <td>0.0</td>
      <td>28961.0</td>
      <td>22664.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>13742.0</td>
      <td>589288.0</td>
      <td>4581.0</td>
      <td>0.0</td>
      <td>107.0</td>
      <td>10641.0</td>
      <td>26961.0</td>
      <td>3328.0</td>
      <td>3236.0</td>
      <td>3940.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 174 columns</p>
</div>



Now, this is not perfect so we transpose this dataframe and add column names.


```python
new_df = pd.DataFrame.transpose(new_df)
new_df.columns = year_list

new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>Y1968</th>
      <th>Y1969</th>
      <th>Y1970</th>
      <th>...</th>
      <th>Y2004</th>
      <th>Y2005</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Afghanistan</td>
      <td>9481.0</td>
      <td>9414.0</td>
      <td>9194.0</td>
      <td>10170.0</td>
      <td>10473.0</td>
      <td>10169.0</td>
      <td>11289.0</td>
      <td>11508.0</td>
      <td>11815.0</td>
      <td>10454.0</td>
      <td>...</td>
      <td>16542.0</td>
      <td>17658.0</td>
      <td>18317.0</td>
      <td>19248.0</td>
      <td>19381.0</td>
      <td>20661.0</td>
      <td>21030.0</td>
      <td>21100.0</td>
      <td>22706.0</td>
      <td>23007.0</td>
    </tr>
    <tr>
      <td>Albania</td>
      <td>1706.0</td>
      <td>1749.0</td>
      <td>1767.0</td>
      <td>1889.0</td>
      <td>1884.0</td>
      <td>1995.0</td>
      <td>2046.0</td>
      <td>2169.0</td>
      <td>2230.0</td>
      <td>2395.0</td>
      <td>...</td>
      <td>6637.0</td>
      <td>6719.0</td>
      <td>6911.0</td>
      <td>6744.0</td>
      <td>7168.0</td>
      <td>7316.0</td>
      <td>7907.0</td>
      <td>8114.0</td>
      <td>8221.0</td>
      <td>8271.0</td>
    </tr>
    <tr>
      <td>Algeria</td>
      <td>7488.0</td>
      <td>7235.0</td>
      <td>6861.0</td>
      <td>7255.0</td>
      <td>7509.0</td>
      <td>7536.0</td>
      <td>7986.0</td>
      <td>8839.0</td>
      <td>9003.0</td>
      <td>9355.0</td>
      <td>...</td>
      <td>48619.0</td>
      <td>49562.0</td>
      <td>51067.0</td>
      <td>49933.0</td>
      <td>50916.0</td>
      <td>57505.0</td>
      <td>60071.0</td>
      <td>65852.0</td>
      <td>69365.0</td>
      <td>72161.0</td>
    </tr>
    <tr>
      <td>Angola</td>
      <td>4834.0</td>
      <td>4775.0</td>
      <td>5240.0</td>
      <td>5286.0</td>
      <td>5527.0</td>
      <td>5677.0</td>
      <td>5833.0</td>
      <td>5685.0</td>
      <td>6219.0</td>
      <td>6460.0</td>
      <td>...</td>
      <td>25541.0</td>
      <td>26696.0</td>
      <td>28247.0</td>
      <td>29877.0</td>
      <td>32053.0</td>
      <td>36985.0</td>
      <td>38400.0</td>
      <td>40573.0</td>
      <td>38064.0</td>
      <td>48639.0</td>
    </tr>
    <tr>
      <td>Antigua and Barbuda</td>
      <td>92.0</td>
      <td>94.0</td>
      <td>105.0</td>
      <td>95.0</td>
      <td>84.0</td>
      <td>73.0</td>
      <td>64.0</td>
      <td>59.0</td>
      <td>68.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>115.0</td>
      <td>110.0</td>
      <td>122.0</td>
      <td>115.0</td>
      <td>114.0</td>
      <td>115.0</td>
      <td>118.0</td>
      <td>113.0</td>
      <td>119.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>



Perfect! Now, we will do some feature engineering.

# First, a new column which indicates mean produce of each state over the given years. Second, a ranking column which ranks countries on the basis of mean produce.


```python
mean_produce = []
for i in range(174):
    mean_produce.append(new_df.iloc[i,:].values.mean())
new_df['Mean_Produce'] = mean_produce

new_df['Rank'] = new_df['Mean_Produce'].rank(ascending=False)

new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>Y1968</th>
      <th>Y1969</th>
      <th>Y1970</th>
      <th>...</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
      <th>Mean_Produce</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Afghanistan</td>
      <td>9481.0</td>
      <td>9414.0</td>
      <td>9194.0</td>
      <td>10170.0</td>
      <td>10473.0</td>
      <td>10169.0</td>
      <td>11289.0</td>
      <td>11508.0</td>
      <td>11815.0</td>
      <td>10454.0</td>
      <td>...</td>
      <td>18317.0</td>
      <td>19248.0</td>
      <td>19381.0</td>
      <td>20661.0</td>
      <td>21030.0</td>
      <td>21100.0</td>
      <td>22706.0</td>
      <td>23007.0</td>
      <td>13003.056604</td>
      <td>69.0</td>
    </tr>
    <tr>
      <td>Albania</td>
      <td>1706.0</td>
      <td>1749.0</td>
      <td>1767.0</td>
      <td>1889.0</td>
      <td>1884.0</td>
      <td>1995.0</td>
      <td>2046.0</td>
      <td>2169.0</td>
      <td>2230.0</td>
      <td>2395.0</td>
      <td>...</td>
      <td>6911.0</td>
      <td>6744.0</td>
      <td>7168.0</td>
      <td>7316.0</td>
      <td>7907.0</td>
      <td>8114.0</td>
      <td>8221.0</td>
      <td>8271.0</td>
      <td>4475.509434</td>
      <td>104.0</td>
    </tr>
    <tr>
      <td>Algeria</td>
      <td>7488.0</td>
      <td>7235.0</td>
      <td>6861.0</td>
      <td>7255.0</td>
      <td>7509.0</td>
      <td>7536.0</td>
      <td>7986.0</td>
      <td>8839.0</td>
      <td>9003.0</td>
      <td>9355.0</td>
      <td>...</td>
      <td>51067.0</td>
      <td>49933.0</td>
      <td>50916.0</td>
      <td>57505.0</td>
      <td>60071.0</td>
      <td>65852.0</td>
      <td>69365.0</td>
      <td>72161.0</td>
      <td>28879.490566</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>Angola</td>
      <td>4834.0</td>
      <td>4775.0</td>
      <td>5240.0</td>
      <td>5286.0</td>
      <td>5527.0</td>
      <td>5677.0</td>
      <td>5833.0</td>
      <td>5685.0</td>
      <td>6219.0</td>
      <td>6460.0</td>
      <td>...</td>
      <td>28247.0</td>
      <td>29877.0</td>
      <td>32053.0</td>
      <td>36985.0</td>
      <td>38400.0</td>
      <td>40573.0</td>
      <td>38064.0</td>
      <td>48639.0</td>
      <td>13321.056604</td>
      <td>68.0</td>
    </tr>
    <tr>
      <td>Antigua and Barbuda</td>
      <td>92.0</td>
      <td>94.0</td>
      <td>105.0</td>
      <td>95.0</td>
      <td>84.0</td>
      <td>73.0</td>
      <td>64.0</td>
      <td>59.0</td>
      <td>68.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>122.0</td>
      <td>115.0</td>
      <td>114.0</td>
      <td>115.0</td>
      <td>118.0</td>
      <td>113.0</td>
      <td>119.0</td>
      <td>83.886792</td>
      <td>172.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



Now, we create another dataframe with items and their total production each year from 1961 to 2013


```python
item_list = list(df['Item'].unique())

item_df = pd.DataFrame()
item_df['Item_Name'] = item_list

for yr in year_list:
    item_produce = []
    for it in item_list:
        item_produce.append(df[yr][df['Item']==it].sum())
    item_df[yr] = item_produce

```


```python
item_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item_Name</th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>Y1968</th>
      <th>Y1969</th>
      <th>...</th>
      <th>Y2004</th>
      <th>Y2005</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Wheat and products</td>
      <td>138829.0</td>
      <td>144643.0</td>
      <td>147325.0</td>
      <td>156273.0</td>
      <td>168822.0</td>
      <td>169832.0</td>
      <td>171469.0</td>
      <td>179530.0</td>
      <td>189658.0</td>
      <td>...</td>
      <td>527394.0</td>
      <td>532263.0</td>
      <td>537279.0</td>
      <td>529271.0</td>
      <td>562239.0</td>
      <td>557245.0</td>
      <td>549926.0</td>
      <td>578179.0</td>
      <td>576597</td>
      <td>587492</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Rice (Milled Equivalent)</td>
      <td>122700.0</td>
      <td>131842.0</td>
      <td>139507.0</td>
      <td>148304.0</td>
      <td>150056.0</td>
      <td>155583.0</td>
      <td>158587.0</td>
      <td>164614.0</td>
      <td>167922.0</td>
      <td>...</td>
      <td>361107.0</td>
      <td>366025.0</td>
      <td>372629.0</td>
      <td>378698.0</td>
      <td>389708.0</td>
      <td>394221.0</td>
      <td>398559.0</td>
      <td>404152.0</td>
      <td>406787</td>
      <td>410880</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Barley and products</td>
      <td>46180.0</td>
      <td>48915.0</td>
      <td>51642.0</td>
      <td>54184.0</td>
      <td>54945.0</td>
      <td>55463.0</td>
      <td>56424.0</td>
      <td>60455.0</td>
      <td>65501.0</td>
      <td>...</td>
      <td>102055.0</td>
      <td>97185.0</td>
      <td>100981.0</td>
      <td>93310.0</td>
      <td>98209.0</td>
      <td>99135.0</td>
      <td>92563.0</td>
      <td>92570.0</td>
      <td>88766</td>
      <td>99452</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Maize and products</td>
      <td>168039.0</td>
      <td>168305.0</td>
      <td>172905.0</td>
      <td>175468.0</td>
      <td>190304.0</td>
      <td>200860.0</td>
      <td>213050.0</td>
      <td>215613.0</td>
      <td>221953.0</td>
      <td>...</td>
      <td>545024.0</td>
      <td>549036.0</td>
      <td>543280.0</td>
      <td>573892.0</td>
      <td>592231.0</td>
      <td>557940.0</td>
      <td>584337.0</td>
      <td>603297.0</td>
      <td>608730</td>
      <td>671300</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Millet and products</td>
      <td>19075.0</td>
      <td>19019.0</td>
      <td>19740.0</td>
      <td>20353.0</td>
      <td>18377.0</td>
      <td>20860.0</td>
      <td>22997.0</td>
      <td>21785.0</td>
      <td>23966.0</td>
      <td>...</td>
      <td>25789.0</td>
      <td>25496.0</td>
      <td>25997.0</td>
      <td>26750.0</td>
      <td>26373.0</td>
      <td>24575.0</td>
      <td>27039.0</td>
      <td>25740.0</td>
      <td>26105</td>
      <td>26346</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>



# Some more feature engineering

This time, we will use the new features to get some good conclusions.

# 1. Total amount of item produced from 1961 to 2013
# 2. Providing a rank to the items to know the most produced item


```python
sum_col = []
for i in range(115):
    sum_col.append(item_df.iloc[i,1:].values.sum())
item_df['Sum'] = sum_col
item_df['Production_Rank'] = item_df['Sum'].rank(ascending=False)

item_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item_Name</th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>Y1968</th>
      <th>Y1969</th>
      <th>...</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
      <th>Sum</th>
      <th>Production_Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Wheat and products</td>
      <td>138829.0</td>
      <td>144643.0</td>
      <td>147325.0</td>
      <td>156273.0</td>
      <td>168822.0</td>
      <td>169832.0</td>
      <td>171469.0</td>
      <td>179530.0</td>
      <td>189658.0</td>
      <td>...</td>
      <td>537279.0</td>
      <td>529271.0</td>
      <td>562239.0</td>
      <td>557245.0</td>
      <td>549926.0</td>
      <td>578179.0</td>
      <td>576597</td>
      <td>587492</td>
      <td>19194671.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Rice (Milled Equivalent)</td>
      <td>122700.0</td>
      <td>131842.0</td>
      <td>139507.0</td>
      <td>148304.0</td>
      <td>150056.0</td>
      <td>155583.0</td>
      <td>158587.0</td>
      <td>164614.0</td>
      <td>167922.0</td>
      <td>...</td>
      <td>372629.0</td>
      <td>378698.0</td>
      <td>389708.0</td>
      <td>394221.0</td>
      <td>398559.0</td>
      <td>404152.0</td>
      <td>406787</td>
      <td>410880</td>
      <td>14475448.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Barley and products</td>
      <td>46180.0</td>
      <td>48915.0</td>
      <td>51642.0</td>
      <td>54184.0</td>
      <td>54945.0</td>
      <td>55463.0</td>
      <td>56424.0</td>
      <td>60455.0</td>
      <td>65501.0</td>
      <td>...</td>
      <td>100981.0</td>
      <td>93310.0</td>
      <td>98209.0</td>
      <td>99135.0</td>
      <td>92563.0</td>
      <td>92570.0</td>
      <td>88766</td>
      <td>99452</td>
      <td>4442742.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Maize and products</td>
      <td>168039.0</td>
      <td>168305.0</td>
      <td>172905.0</td>
      <td>175468.0</td>
      <td>190304.0</td>
      <td>200860.0</td>
      <td>213050.0</td>
      <td>215613.0</td>
      <td>221953.0</td>
      <td>...</td>
      <td>543280.0</td>
      <td>573892.0</td>
      <td>592231.0</td>
      <td>557940.0</td>
      <td>584337.0</td>
      <td>603297.0</td>
      <td>608730</td>
      <td>671300</td>
      <td>19960640.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Millet and products</td>
      <td>19075.0</td>
      <td>19019.0</td>
      <td>19740.0</td>
      <td>20353.0</td>
      <td>18377.0</td>
      <td>20860.0</td>
      <td>22997.0</td>
      <td>21785.0</td>
      <td>23966.0</td>
      <td>...</td>
      <td>25997.0</td>
      <td>26750.0</td>
      <td>26373.0</td>
      <td>24575.0</td>
      <td>27039.0</td>
      <td>25740.0</td>
      <td>26105</td>
      <td>26346</td>
      <td>1225400.0</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>



# Now, we find the most produced food items in the last half-century


```python
item_df['Item_Name'][item_df['Production_Rank'] < 11.0].sort_values()
```




    56    Cereals - Excluding Beer
    65     Fruits - Excluding Wine
    3           Maize and products
    53     Milk - Excluding Butter
    6        Potatoes and products
    1     Rice (Milled Equivalent)
    57               Starchy Roots
    64                  Vegetables
    27           Vegetables, Other
    0           Wheat and products
    Name: Item_Name, dtype: object



So, cereals, fruits and maize are the most produced items in the last 50 years

# Food and feed plot for most produced items 


```python
sns.factorplot("Item", data=df[(df['Item']=='Wheat and products') | (df['Item']=='Rice (Milled Equivalent)') | (df['Item']=='Maize and products') | (df['Item']=='Potatoes and products') | (df['Item']=='Vegetables, Other') | (df['Item']=='Milk - Excluding Butter') | (df['Item']=='Cereals - Excluding Beer') | (df['Item']=='Starchy Roots') | (df['Item']=='Vegetables') | (df['Item']=='Fruits - Excluding Wine')], kind="count", hue="Element", size=20, aspect=.8)
plt.show()
```


![png](/images/output_25_0.png)


# Now, we plot a heatmap of correlation of produce in difference years


```python
year_df = df.iloc[:,10:]
fig, ax = plt.subplots(figsize=(16,10))
sns.heatmap(year_df.corr(), ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe3a58530b8>




![png](/images/output_27_1.png)


So, we gather that a given year's production is more similar to its immediate previous and immediate following years.


```python
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,10))
ax1.set(xlabel='Y1968', ylabel='Y1961')
ax2.set(xlabel='Y1968', ylabel='Y1963')
ax3.set(xlabel='Y1968', ylabel='Y1986')
ax4.set(xlabel='Y1968', ylabel='Y2013')
sns.jointplot(x="Y1968", y="Y1961", data=df, kind="reg", ax=ax1)
sns.jointplot(x="Y1968", y="Y1963", data=df, kind="reg", ax=ax2)
sns.jointplot(x="Y1968", y="Y1986", data=df, kind="reg", ax=ax3)
sns.jointplot(x="Y1968", y="Y2013", data=df, kind="reg", ax=ax4)
plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
```


![png](/images/output_29_0.png)


# Heatmap of production of food items over years

This will detect the items whose production has drastically increased over the years


```python
new_item_df = item_df.drop(["Item_Name","Sum","Production_Rank"], axis = 1)
fig, ax = plt.subplots(figsize=(12,24))
sns.heatmap(new_item_df,ax=ax)
ax.set_yticklabels(item_df.Item_Name.values[::-1])
plt.show()
```


![png](/images/output_31_00.png)


There is considerable growth in production of Palmkernel oil, Meat/Aquatic animals, ricebran oil, cottonseed, seafood, offals, roots, poultry meat, mutton, bear, cocoa, coffee and soyabean oil.
There has been exceptional growth in production of onions, cream, sugar crops, treenuts, butter/ghee and to some extent starchy roots.

Now, we look at clustering.

# What is clustering?
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.

# Today, we will form clusters to classify countries based on productivity scale

For this, we will use k-means clustering algorithm.
# K-means clustering
(Source [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm) )
![http://gdurl.com/5BbP](http://gdurl.com/5BbP)

This is the data we will use.


```python
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>Y1968</th>
      <th>Y1969</th>
      <th>Y1970</th>
      <th>...</th>
      <th>Y2006</th>
      <th>Y2007</th>
      <th>Y2008</th>
      <th>Y2009</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
      <th>Mean_Produce</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Afghanistan</td>
      <td>9481.0</td>
      <td>9414.0</td>
      <td>9194.0</td>
      <td>10170.0</td>
      <td>10473.0</td>
      <td>10169.0</td>
      <td>11289.0</td>
      <td>11508.0</td>
      <td>11815.0</td>
      <td>10454.0</td>
      <td>...</td>
      <td>18317.0</td>
      <td>19248.0</td>
      <td>19381.0</td>
      <td>20661.0</td>
      <td>21030.0</td>
      <td>21100.0</td>
      <td>22706.0</td>
      <td>23007.0</td>
      <td>13003.056604</td>
      <td>69.0</td>
    </tr>
    <tr>
      <td>Albania</td>
      <td>1706.0</td>
      <td>1749.0</td>
      <td>1767.0</td>
      <td>1889.0</td>
      <td>1884.0</td>
      <td>1995.0</td>
      <td>2046.0</td>
      <td>2169.0</td>
      <td>2230.0</td>
      <td>2395.0</td>
      <td>...</td>
      <td>6911.0</td>
      <td>6744.0</td>
      <td>7168.0</td>
      <td>7316.0</td>
      <td>7907.0</td>
      <td>8114.0</td>
      <td>8221.0</td>
      <td>8271.0</td>
      <td>4475.509434</td>
      <td>104.0</td>
    </tr>
    <tr>
      <td>Algeria</td>
      <td>7488.0</td>
      <td>7235.0</td>
      <td>6861.0</td>
      <td>7255.0</td>
      <td>7509.0</td>
      <td>7536.0</td>
      <td>7986.0</td>
      <td>8839.0</td>
      <td>9003.0</td>
      <td>9355.0</td>
      <td>...</td>
      <td>51067.0</td>
      <td>49933.0</td>
      <td>50916.0</td>
      <td>57505.0</td>
      <td>60071.0</td>
      <td>65852.0</td>
      <td>69365.0</td>
      <td>72161.0</td>
      <td>28879.490566</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>Angola</td>
      <td>4834.0</td>
      <td>4775.0</td>
      <td>5240.0</td>
      <td>5286.0</td>
      <td>5527.0</td>
      <td>5677.0</td>
      <td>5833.0</td>
      <td>5685.0</td>
      <td>6219.0</td>
      <td>6460.0</td>
      <td>...</td>
      <td>28247.0</td>
      <td>29877.0</td>
      <td>32053.0</td>
      <td>36985.0</td>
      <td>38400.0</td>
      <td>40573.0</td>
      <td>38064.0</td>
      <td>48639.0</td>
      <td>13321.056604</td>
      <td>68.0</td>
    </tr>
    <tr>
      <td>Antigua and Barbuda</td>
      <td>92.0</td>
      <td>94.0</td>
      <td>105.0</td>
      <td>95.0</td>
      <td>84.0</td>
      <td>73.0</td>
      <td>64.0</td>
      <td>59.0</td>
      <td>68.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>110.0</td>
      <td>122.0</td>
      <td>115.0</td>
      <td>114.0</td>
      <td>115.0</td>
      <td>118.0</td>
      <td>113.0</td>
      <td>119.0</td>
      <td>83.886792</td>
      <td>172.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>




```python
X = new_df.iloc[:,:-2].values

X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
X.columns = year_list
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-23-ebc6771564a7> in <module>
          2 
          3 X = pd.DataFrame(X)
    ----> 4 X = X.convert_objects(convert_numeric=True)
          5 X.columns = year_list
    

    /opt/conda/lib/python3.6/site-packages/pandas/core/generic.py in __getattr__(self, name)
       5177             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5178                 return self[name]
    -> 5179             return object.__getattribute__(self, name)
       5180 
       5181     def __setattr__(self, name, value):
    

    AttributeError: 'DataFrame' object has no attribute 'convert_objects'


# Elbow method to select number of clusters
This method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion". This "elbow" cannot always be unambiguously identified. Percentage of variance explained is the ratio of the between-group variance to the total variance, also known as an F-test. A slight variation of this method plots the curvature of the within group variance.
# Basically, number of clusters = the x-axis value of the point that is the corner of the "elbow"(the plot looks often looks like an elbow)


```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

As the elbow corner coincides with x=2, we will have to form **2 clusters**. Personally, I would have liked to select 3 to 4 clusters. But trust me, only selecting 2 clusters can lead to best results.
Now, we apply k-means algorithm.


```python
kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)

X = X.as_matrix(columns=None)
```

Now, let's visualize the results.


```python
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='Others')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='China(mainland),USA,India')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of countries by Productivity')
plt.legend()
plt.show()
```

So, the blue cluster represents China(Mainland), USA and India while the red cluster represents all the other countries.
This result was highly probable. Just take a look at the plot of cell 3 above. See how China, USA and India stand out. That has been observed here in clustering too.

You should try this algorithm for 3 or 4 clusters. Looking at the distribution, you will realise why 2 clusters is the best choice for the given data

This is not the end! More is yet to come.
