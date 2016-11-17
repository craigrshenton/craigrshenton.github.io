---
layout: post
title: "Data smarts in python- 1.3 pivot table"
tags:
    - python
    - notebook
---

Continuing from chapter 1. Make a pivot table for revenue per item / category

**In [1]:**

```python
# revenue = price * number of sales
pivot = pandas.pivot_table(df_sales, index=["Item"], values=["Price"], columns=["Category"], aggfunc=np.sum, fill_value='')
pivot.index.name = None
pivot.columns = pivot.columns.get_level_values(1) # sets cols to product categories
pivot
```

<div>
<table rules="groups">
  <thead>
    <tr>
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

<!--more-->

Make a pivot table for number of sales per item

**In [2]:**

```python
pivot = pandas.pivot_table(df_sales, index=["Item"], values=["Price"], aggfunc=len) # len == 'count of price'
pivot.columns = ['Count'] # renames col
pivot.index.name = None # removes intex title which is not needed
pivot
```

<div>
<table rules="groups">
  <thead>
    <tr>
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

