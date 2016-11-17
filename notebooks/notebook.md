---
layout: post
title: "notebook"
tags:
    - python
    - notebook
---
##### 1.0 Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch01.zip, Concessions.xlsx

**In [22]:**

{% highlight python %}
# code written in python_3. (for py_2.7 users some changes may be required)

import pandas # load pandas dataframe lib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

# find path to your Concessions.xlsx
# df = short for dataframe == excel worksheet
# zero indexing in python, so first worksheet = 0
df_sales = pandas.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch01_complete/Concessions.xlsx','rb'), sheetname=0) 
df_sales = df_sales.iloc[0:, 0:4]
df_sales.head() # use .head() to just show top 4 results
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item</th>
      <th>Category</th>
      <th>Price</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beer</td>
      <td>Beverages</td>
      <td>4.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hamburger</td>
      <td>Hot Food</td>
      <td>3.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Popcorn</td>
      <td>Hot Food</td>
      <td>5.0</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pizza</td>
      <td>Hot Food</td>
      <td>2.0</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bottled Water</td>
      <td>Beverages</td>
      <td>3.0</td>
      <td>0.833333</td>
    </tr>
  </tbody>
</table>
</div>



**In [5]:**

{% highlight python %}
df_sales.dtypes # explore the dataframe
{% endhighlight %}




    Item         object
    Category     object
    Price       float64
    Profit      float64
    dtype: object



**In [6]:**

{% highlight python %}
df_sales['Item'].head() # how to select a col
{% endhighlight %}




    0             Beer
    1        Hamburger
    2          Popcorn
    3            Pizza
    4    Bottled Water
    Name: Item, dtype: object



**In [7]:**

{% highlight python %}
df_sales['Price'].describe() # basic stats
{% endhighlight %}




    count    199.000000
    mean       2.829146
    std        0.932551
    min        1.500000
    25%        2.000000
    50%        3.000000
    75%        3.000000
    max        5.000000
    Name: Price, dtype: float64



##### 1.2 Calculate Actual Profit

**In [8]:**

{% highlight python %}
df_sales = df_sales.assign(Actual_Profit = df_sales['Price']*df_sales['Profit']) # adds new col
df_sales.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item</th>
      <th>Category</th>
      <th>Price</th>
      <th>Profit</th>
      <th>Actual_Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beer</td>
      <td>Beverages</td>
      <td>4.0</td>
      <td>0.500000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hamburger</td>
      <td>Hot Food</td>
      <td>3.0</td>
      <td>0.666667</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Popcorn</td>
      <td>Hot Food</td>
      <td>5.0</td>
      <td>0.800000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pizza</td>
      <td>Hot Food</td>
      <td>2.0</td>
      <td>0.250000</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bottled Water</td>
      <td>Beverages</td>
      <td>3.0</td>
      <td>0.833333</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.3 Load data from 'Calories' worksheet and plot

**In [9]:**

{% highlight python %}
# find path to your Concessions.xlsx 
df_cals = pandas.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch01_complete/Concessions.xlsx','rb'), sheetname=1) 
df_cals = df_cals.iloc[0:14, 0:2] # take data from 'Calories' worksheet
df_cals.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beer</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bottled Water</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chocolate Bar</td>
      <td>255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chocolate Dipped Cone</td>
      <td>300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gummy Bears</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>



**In [15]:**

{% highlight python %}
df_cals = df_cals.set_index('Item') # index df by items
# Items ranked by calories = .sort_values(by='Calories',ascending=True) 
# rot = axis rotation
ax = df_cals.sort_values(by='Calories',ascending=True).plot(kind='bar', title ="Calories",figsize=(15,5),legend=False, fontsize=10, alpha=0.75, rot=20,)
plt.xlabel("") # no x-axis lable
plt.show()
{% endhighlight %}


![png]({{ site.baseurl }}/notebooks/notebook_files/notebook_9_0.png)


##### 1.4 add calorie data to sales worksheet

**In [42]:**

{% highlight python %}
df_sales = df_sales.assign(Calories=df_sales['Item'].map(df_cals['Calories'])) # map num calories from df_cals per item in df_sales (==Vlookup)
df_sales.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item</th>
      <th>Category</th>
      <th>Price</th>
      <th>Profit</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beer</td>
      <td>Beverages</td>
      <td>4.0</td>
      <td>0.500000</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hamburger</td>
      <td>Hot Food</td>
      <td>3.0</td>
      <td>0.666667</td>
      <td>320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Popcorn</td>
      <td>Hot Food</td>
      <td>5.0</td>
      <td>0.800000</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pizza</td>
      <td>Hot Food</td>
      <td>2.0</td>
      <td>0.250000</td>
      <td>480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bottled Water</td>
      <td>Beverages</td>
      <td>3.0</td>
      <td>0.833333</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.5 pivot table: number of sales per item

**In [96]:**

{% highlight python %}
pivot = pandas.pivot_table(df_sales, index=["Item"], values=["Price"], aggfunc=len) # len == 'count of price'
pivot.columns = ['Count'] # renames col
pivot.index.name = None # removes intex title which is not needed
pivot
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Beer</th>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Bottled Water</th>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Chocolate Bar</th>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Chocolate Dipped Cone</th>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Gummy Bears</th>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Hamburger</th>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Hot Dog</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Ice Cream Sandwich</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Licorice Rope</th>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Nachos</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Pizza</th>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Popcorn</th>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Popsicle</th>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Soda</th>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.6 pivot table: revenue per item / category

**In [108]:**

{% highlight python %}
# revenue = price * number of sales
pivot = pandas.pivot_table(df_sales, index=["Item"], values=["Price"], columns=["Category"], aggfunc=np.sum, fill_value='')
pivot.index.name = None
pivot.columns = pivot.columns.get_level_values(1) # sets cols to product categories
pivot
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Beverages</th>
      <th>Candy</th>
      <th>Frozen Treats</th>
      <th>Hot Food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Beer</th>
      <td>80</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Bottled Water</th>
      <td>39</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Chocolate Bar</th>
      <td></td>
      <td>26</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Chocolate Dipped Cone</th>
      <td></td>
      <td></td>
      <td>33</td>
      <td></td>
    </tr>
    <tr>
      <th>Gummy Bears</th>
      <td></td>
      <td>28</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Hamburger</th>
      <td></td>
      <td></td>
      <td></td>
      <td>48</td>
    </tr>
    <tr>
      <th>Hot Dog</th>
      <td></td>
      <td></td>
      <td></td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>Ice Cream Sandwich</th>
      <td></td>
      <td></td>
      <td>30</td>
      <td></td>
    </tr>
    <tr>
      <th>Licorice Rope</th>
      <td></td>
      <td>26</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Nachos</th>
      <td></td>
      <td></td>
      <td></td>
      <td>45</td>
    </tr>
    <tr>
      <th>Pizza</th>
      <td></td>
      <td></td>
      <td></td>
      <td>34</td>
    </tr>
    <tr>
      <th>Popcorn</th>
      <td></td>
      <td></td>
      <td></td>
      <td>80</td>
    </tr>
    <tr>
      <th>Popsicle</th>
      <td></td>
      <td></td>
      <td>39</td>
      <td></td>
    </tr>
    <tr>
      <th>Soda</th>
      <td>32.5</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



**In [None]:**

{% highlight python %}

{% endhighlight %}
