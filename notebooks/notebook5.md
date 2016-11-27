---
layout: post
title: "notebook5"
tags:
    - python
    - notebook
---
Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch02.zip, WineKMC.xlsx

**In [1]:**

{% highlight python %}
# code written in py_3.0

import pandas as pd
import numpy as np

df_sales = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch02/WineKMC.xlsx','rb'), sheetname=1)
df_sales.columns = ['name', 'offer']
df_sales.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>offer</th>
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



**In [2]:**

{% highlight python %}
# get list unique customer names
names = df_sales.name.unique()
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



**In [87]:**

{% highlight python %}
# make dataframe of customer names
df_names = pd.DataFrame({"name": names})
id = df_names.index+1 # give each name a unique id number
id = id.unique()
id
{% endhighlight %}




    Int64Index([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
                 27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
                 53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
                 66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
                 79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
                 92,  93,  94,  95,  96,  97,  98,  99, 100],
               dtype='int64')



**In [4]:**

{% highlight python %}
id_dict = dict(zip(names, id))
df_sales['id']=df_sales.name.map(id_dict)
df_sales.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>offer</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Smith</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Johnson</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Johnson</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Johnson</td>
      <td>26</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**In [5]:**

{% highlight python %}
pivot = pd.pivot_table(df_sales, index=["offer"], columns=["id"], aggfunc=len, fill_value='0')
pivot.index.name = None
pivot.columns = pivot.columns.get_level_values(1) # sets cols to product categories
X = pivot.as_matrix()
X = np.matrix(X)
X = X.astype(int)
X
{% endhighlight %}




    matrix([[0, 0, 0, ..., 1, 0, 1],
            [1, 0, 0, ..., 0, 0, 1],
            [0, 0, 0, ..., 0, 0, 0],
            ..., 
            [0, 0, 0, ..., 1, 0, 1],
            [0, 0, 1, ..., 0, 1, 1],
            [0, 0, 0, ..., 0, 0, 0]])



**In [54]:**

{% highlight python %}
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

dist_out = 1-pairwise_distances(X.T, metric="cosine")
dist_out
{% endhighlight %}




    array([[ 1.        ,  0.40824829,  0.        , ...,  0.        ,
             0.        ,  0.26726124],
           [ 0.40824829,  1.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        , ...,  0.25819889,
             0.57735027,  0.43643578],
           ..., 
           [ 0.        ,  0.        ,  0.25819889, ...,  1.        ,
             0.2236068 ,  0.6761234 ],
           [ 0.        ,  0.        ,  0.57735027, ...,  0.2236068 ,
             1.        ,  0.37796447],
           [ 0.26726124,  0.        ,  0.43643578, ...,  0.6761234 ,
             0.37796447,  1.        ]])



**In [56]:**

{% highlight python %}
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_numpy_matrix(dist_out)
G.graph['name']='cosine similarity graph'

# create network layout for visualizations
pos = nx.spring_layout(G)

nx.draw(G, pos, node_size=50)
print(nx.info(G))
plt.show()
{% endhighlight %}

    Name: cosine similarity graph
    Type: Graph
    Number of nodes: 100
    Number of edges: 1575
    Average degree:  31.5000
    


![png]({{ site.baseurl }}/notebooks/notebook5_files/notebook5_7_1.png)


**In [69]:**

{% highlight python %}
r_hood = dist_out < 0.5  # filter out low similarity edges
dist_out[r_hood] = 0      # low values set to 0
G = nx.from_numpy_matrix(dist_out)
G.graph['name']='r-filtered similarity graph'

# create network layout for visualizations
pos = nx.spring_layout(G)

nx.draw(G, pos, node_size=50)
print(nx.info(G))
plt.show() # show filtered graph
{% endhighlight %}

    Name: r-filtered similarity graph
    Type: Graph
    Number of nodes: 100
    Number of edges: 378
    Average degree:   7.5600
    


![png]({{ site.baseurl }}/notebooks/notebook5_files/notebook5_8_1.png)


**In [76]:**

{% highlight python %}
import community

# find communities
part = community.best_partition(G)
G.graph['name']='community graph'

# create network layout for visualizations
pos = nx.spring_layout(G)

# plot and color nodes using community structure
community_num = [part.get(node) for node in G.nodes()]
nx.draw(G, pos, cmap = plt.get_cmap("jet"), node_color = community_num, node_size = 50)
print(nx.info(G))
plt.show()
{% endhighlight %}

    Name: community graph
    Type: Graph
    Number of nodes: 100
    Number of edges: 378
    Average degree:   7.5600
    


![png]({{ site.baseurl }}/notebooks/notebook5_files/notebook5_9_1.png)


**In [83]:**

{% highlight python %}
# find modularity
mod = community.modularity(part,G)
print("modularity:", mod)
{% endhighlight %}

    modularity: 0.7116192085619842
    

**In [131]:**

{% highlight python %}
community_num = [x+1 for x in community_num] # non-zero indexing for commmunity list
community_dict = dict(zip(names, community_num))
df_sales['community']=df_sales.name.map(community_dict) # map communities to sales
df_sales = df_sales.drop('id', 1) # don't need this now
df_sales.head() # note: first five all in same community
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>offer</th>
      <th>community</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Smith</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Johnson</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Johnson</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Johnson</td>
      <td>26</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**In [132]:**

{% highlight python %}
from collections import Counter

count_dict = dict(zip(df_sales['community'], df_sales['offer']))
count_list = Counter(count_dict)
df_communities = pd.DataFrame(sorted(count_list.most_common()))
df_communities.columns = ['Community', 'Most Common Offer']
df_communities
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Community</th>
      <th>Most Common Offer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>29</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>31</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>30</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>31</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>31</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>32</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



**In [133]:**

{% highlight python %}
df_offers = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch02/WineKMC.xlsx','rb'), sheetname=0) 
df_communities['Origin']=df_offers['Offer #'].map(df_offers.Origin) # map offer region to communities
df_communities['Varietal']=df_offers['Offer #'].map(df_offers.Varietal) # map offer name to communities
df_communities
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Community</th>
      <th>Most Common Offer</th>
      <th>Origin</th>
      <th>Varietal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>26</td>
      <td>France</td>
      <td>Pinot Noir</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Oregon</td>
      <td>Espumante</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>30</td>
      <td>France</td>
      <td>Champagne</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>New Zealand</td>
      <td>Cabernet Sauvignon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
      <td>Chile</td>
      <td>Prosecco</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>5</td>
      <td>Australia</td>
      <td>Prosecco</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>29</td>
      <td>South Africa</td>
      <td>Espumante</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>31</td>
      <td>Chile</td>
      <td>Chardonnay</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>30</td>
      <td>California</td>
      <td>Prosecco</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>31</td>
      <td>France</td>
      <td>Champagne</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>31</td>
      <td>Australia</td>
      <td>Prosecco</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>31</td>
      <td>Chile</td>
      <td>Merlot</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>32</td>
      <td>Chile</td>
      <td>Merlot</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>32</td>
      <td>Italy</td>
      <td>Cabernet Sauvignon</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>4</td>
      <td>California</td>
      <td>Merlot</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>21</td>
      <td>Germany</td>
      <td>Pinot Noir</td>
    </tr>
  </tbody>
</table>
</div>



**In [None]:**

{% highlight python %}

{% endhighlight %}
