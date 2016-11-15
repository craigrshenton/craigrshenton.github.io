---
layout: post
title: "Data smarts in python- 1.0 basics"
tags:
    - ipython
    - notebook
---

Python translation of John W. Foreman's 2014 book "Data Smart: Using Data Science to Transform Information into Insight".

**1.0** Download data from [http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch01.zip](http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch01.zip), extract Concessions.xlsx

**In [1]:**

```python
# python_3.

import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas # load pandas dataframe lib

# df = short for dataframe == excel worksheet
# ...find path to your local Concessions.xlsx
df_sales = pandas.read_excel(open('.../Concessions.xlsx','rb'), sheetname=0)
# zero indexing in python, so first worksheet = 0
df_sales = df_sales.iloc[0:, 0:4]
df_sales.head() # use .head() to just show top 4 results
```
<div>
 <table rules="groups"> 		
   <thead>		
     <tr>		
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

<!--more-->

**In [2]:**

```python
df_sales.dtypes # explore the dataframe
```




    Item         object
    Category     object
    Price       float64
    Profit      float64
    dtype: object



**In [3]:**

```python
df_sales['Item'].head() # how to select a col
```




    0             Beer
    1        Hamburger
    2          Popcorn
    3            Pizza
    4    Bottled Water
    Name: Item, dtype: object



**In [4]:**

```python
df_sales['Price'].describe() # basic stats
```




    count    199.000000
    mean       2.829146
    std        0.932551
    min        1.500000
    25%        2.000000
    50%        3.000000
    75%        3.000000
    max        5.000000
    Name: Price, dtype: float64



**1.2** Calculate Actual Profit

**In [5]:**

```python
# add new col == price*profit
df_sales = df_sales.assign(Actual_Profit = df_sales['Price']*df_sales['Profit'])
df_sales.head()
```

<div>
<table rules="groups">
  <thead>
    <tr>
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
