---
layout: post
title: "Data smarts in python- 1.2 vlookup"
tags:
    - ipython
    - notebook
---

Continuing from chapter 1. Load data from the 'Calories' worksheet

**In [1]:**

```python
# find path to your Concessions.xlsx
df_cals = pandas.read_excel(open('.../Concessions.xlsx','rb'), sheetname=1)
df_cals = df_cals.iloc[0:14, 0:2] # take data from 'Calories' worksheet
df_cals.head()
```

<div>
<table rules="groups">
  <thead>
    <tr>
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

<!--more-->

Add calorie data to the sales worksheet using panadas *series*.map. This is the equivalent of *Vlookup* in excel.

**In [2]:**

```python
# map num calories from df_cals per item in df_sales (==Vlookup)
df_sales = df_sales.assign(Calories=df_sales['Item'].map(df_cals['Calories']))
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
      <td>2.0</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hamburger</td>
      <td>Hot Food</td>
      <td>3.0</td>
      <td>0.666667</td>
      <td>2.0</td>
      <td>320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Popcorn</td>
      <td>Hot Food</td>
      <td>5.0</td>
      <td>0.800000</td>
      <td>4.0</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pizza</td>
      <td>Hot Food</td>
      <td>2.0</td>
      <td>0.250000</td>
      <td>0.5</td>
      <td>480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bottled Water</td>
      <td>Beverages</td>
      <td>3.0</td>
      <td>0.833333</td>
      <td>2.5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
