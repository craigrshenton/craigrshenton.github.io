---
layout: post
title: "notebook1"
tags:
    - python
    - notebook
---
Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch02.zip, WineKMC.xlsx

**In [85]:**

{% highlight python %}
# code written in py_3.0

import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# find path to your WineKMC.xlsx
df_offers = pandas.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch02/WineKMC.xlsx','rb'), sheetname=0) 
df_offers.head() # use .head() to just show top 5 results
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Offer #</th>
      <th>Campaign</th>
      <th>Varietal</th>
      <th>Minimum Qty (kg)</th>
      <th>Discount (%)</th>
      <th>Origin</th>
      <th>Past Peak</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>January</td>
      <td>Malbec</td>
      <td>72</td>
      <td>56</td>
      <td>France</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>January</td>
      <td>Pinot Noir</td>
      <td>72</td>
      <td>17</td>
      <td>France</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>February</td>
      <td>Espumante</td>
      <td>144</td>
      <td>32</td>
      <td>Oregon</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>February</td>
      <td>Champagne</td>
      <td>72</td>
      <td>48</td>
      <td>France</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>February</td>
      <td>Cabernet Sauvignon</td>
      <td>144</td>
      <td>44</td>
      <td>New Zealand</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



**In [4]:**

{% highlight python %}
df_sales = pandas.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch02/WineKMC.xlsx','rb'), sheetname=1) 
df_sales.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Last Name</th>
      <th>Offer #</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Smith</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Johnson</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Johnson</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Johnson</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



**In [19]:**

{% highlight python %}
pivot = pandas.pivot_table(df_sales, index=["Offer #"], columns=["Customer Last Name"], aggfunc=len, fill_value='0')
#pivot.index.name = None
#pivot.columns = pivot.columns.get_level_values(1) # sets cols to product categories
pivot.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Customer Last Name</th>
      <th>Adams</th>
      <th>Allen</th>
      <th>Anderson</th>
      <th>Bailey</th>
      <th>Baker</th>
      <th>Barnes</th>
      <th>Bell</th>
      <th>Bennett</th>
      <th>Brooks</th>
      <th>Brown</th>
      <th>...</th>
      <th>Turner</th>
      <th>Walker</th>
      <th>Ward</th>
      <th>Watson</th>
      <th>White</th>
      <th>Williams</th>
      <th>Wilson</th>
      <th>Wood</th>
      <th>Wright</th>
      <th>Young</th>
    </tr>
    <tr>
      <th>Offer #</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>



**In [64]:**

{% highlight python %}
# convert it to a numpy matrix
X = pivot.as_matrix()
X = np.matrix(X)

# take the transpose of x
X = X.T
{% endhighlight %}

**In [88]:**

{% highlight python %}
kmeans = KMeans(n_clusters=4, random_state=10).fit_predict(X) # seed of 10 for reproducibility.

kmeans
{% endhighlight %}




    array([2, 1, 3, 2, 1, 1, 3, 2, 1, 2, 0, 3, 2, 0, 0, 3, 0, 3, 2, 1, 2, 1, 1,
           0, 3, 1, 1, 0, 1, 3, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 3, 3, 1, 0, 2, 1,
           2, 1, 0, 0, 1, 1, 3, 3, 1, 2, 3, 1, 2, 1, 1, 1, 0, 2, 2, 3, 3, 1, 1,
           1, 1, 1, 1, 2, 1, 2, 3, 1, 1, 3, 0, 0, 1, 3, 2, 2, 2, 0, 0, 1, 1, 2,
           1, 2, 1, 1, 2, 1, 1, 0])



**In [72]:**

{% highlight python %}
# get list unique customer names
names = df_sales["Customer Last Name"].unique()
names
{% endhighlight %}




    array(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis',
           'Garcia', 'Rodriguez', 'Wilson', 'Martinez', 'Anderson', 'Taylor',
           'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson',
           'White', 'Lopez', 'Lee', 'Gonzalez', 'Harris', 'Clark', 'Lewis',
           'Robinson', 'Walker', 'Perez', 'Hall', 'Young', 'Allen', 'Sanchez',
           'Wright', 'King', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson',
           'Hill', 'Ramirez', 'Campbell', 'Mitchell', 'Roberts', 'Carter',
           'Phillips', 'Evans', 'Turner', 'Torres', 'Parker', 'Collins',
           'Edwards', 'Stewart', 'Flores', 'Morris', 'Nguyen', 'Murphy',
           'Rivera', 'Cook', 'Rogers', 'Morgan', 'Peterson', 'Cooper', 'Reed',
           'Bailey', 'Bell', 'Gomez', 'Kelly', 'Howard', 'Ward', 'Cox', 'Diaz',
           'Richardson', 'Wood', 'Watson', 'Brooks', 'Bennett', 'Gray',
           'James', 'Reyes', 'Cruz', 'Hughes', 'Price', 'Myers', 'Long',
           'Foster', 'Sanders', 'Ross', 'Morales', 'Powell', 'Sullivan',
           'Russell', 'Ortiz', 'Jenkins', 'Gutierrez', 'Perry', 'Butler',
           'Barnes', 'Fisher'], dtype=object)



**In [83]:**

{% highlight python %}
# make dataframe of customer names
df_names = pandas.DataFrame({"Customer Last Name": names})

# add list clusters customers belong to
df_names = df_names.assign(Cluster = kmeans)
df_names.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Last Name</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Smith</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Johnson</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Williams</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brown</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jones</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



**In [90]:**

{% highlight python %}
range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    # initialize kmeans for each n clusters between 2--6
    kmeans = KMeans(n_clusters=n_clusters, random_state=10) # seed of 10 for reproducibility.
    cluster_labels = kmeans.fit_predict(X)

    # silhouette_score for n clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
{% endhighlight %}

    For n_clusters = 2 The average silhouette_score is : 0.0936557328349
    For n_clusters = 3 The average silhouette_score is : 0.118899428636
    For n_clusters = 4 The average silhouette_score is : 0.123470539196
    For n_clusters = 5 The average silhouette_score is : 0.14092516242
    For n_clusters = 6 The average silhouette_score is : 0.137179893911
    For n_clusters = 7 The average silhouette_score is : 0.116109245662
    

kmeans with 5 clusters is optimal for this dataset

**In [None]:**

{% highlight python %}

{% endhighlight %}
