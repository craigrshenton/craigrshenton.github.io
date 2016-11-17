---
layout: post
title: "Data smarts in python- 1.4 linear programming"
tags:
    - python
    - notebook
---

Continuing from chapter 1. Find the minimum number of items to order that total 2400 calories

**In [1]:**

```python
# set up decision variables
items = df_cals.index.tolist()
items
```




    ['Beer',
     'Bottled Water',
     'Chocolate Bar',
     'Chocolate Dipped Cone',
     'Gummy Bears',
     'Hamburger',
     'Hot Dog',
     'Ice Cream Sandwich',
     'Licorice Rope',
     'Nachos',
     'Pizza',
     'Popcorn',
     'Popsicle',
     'Soda']



**In [2]:**

```python
cost = dict(zip(df_cals.index, df_cals.Calories)) # calarific cost of each item
cost
```




    {'Beer': 200,
     'Bottled Water': 0,
     'Chocolate Bar': 255,
     'Chocolate Dipped Cone': 300,
     'Gummy Bears': 300,
     'Hamburger': 320,
     'Hot Dog': 265,
     'Ice Cream Sandwich': 240,
     'Licorice Rope': 280,
     'Nachos': 560,
     'Pizza': 480,
     'Popcorn': 500,
     'Popsicle': 150,
     'Soda': 120}

<!--more-->

**In [3]:**

```python
from pulp import *
# create the LinProg object, set up as a minimisation problem
prob = pulp.LpProblem('Diet', pulp.LpMinimize)

vars = LpVariable.dicts("Number of",items, lowBound = 0, cat='Integer')
# Obj Func
prob += lpSum([cost[c]*vars[c] for c in items])

prob += sum(vars[c] for c in items)

# add constraint representing demand for soldiers
prob += (lpSum([cost[c]*vars[c] for c in items]) == 2400)

print(prob)
```

    Diet:
    MINIMIZE
    1*Number_of_Beer + 1*Number_of_Bottled_Water + 1*Number_of_Chocolate_Bar + 1*Number_of_Chocolate_Dipped_Cone + 1*Number_of_Gummy_Bears + 1*Number_of_Hamburger + 1*Number_of_Hot_Dog + 1*Number_of_Ice_Cream_Sandwich + 1*Number_of_Licorice_Rope + 1*Number_of_Nachos + 1*Number_of_Pizza + 1*Number_of_Popcorn + 1*Number_of_Popsicle + 1*Number_of_Soda + 0
    SUBJECT TO
    _C1: 200 Number_of_Beer + 255 Number_of_Chocolate_Bar
     + 300 Number_of_Chocolate_Dipped_Cone + 300 Number_of_Gummy_Bears
     + 320 Number_of_Hamburger + 265 Number_of_Hot_Dog
     + 240 Number_of_Ice_Cream_Sandwich + 280 Number_of_Licorice_Rope
     + 560 Number_of_Nachos + 480 Number_of_Pizza + 500 Number_of_Popcorn
     + 150 Number_of_Popsicle + 120 Number_of_Soda = 2400

    VARIABLES
    0 <= Number_of_Beer Integer
    0 <= Number_of_Bottled_Water Integer
    0 <= Number_of_Chocolate_Bar Integer
    0 <= Number_of_Chocolate_Dipped_Cone Integer
    0 <= Number_of_Gummy_Bears Integer
    0 <= Number_of_Hamburger Integer
    0 <= Number_of_Hot_Dog Integer
    0 <= Number_of_Ice_Cream_Sandwich Integer
    0 <= Number_of_Licorice_Rope Integer
    0 <= Number_of_Nachos Integer
    0 <= Number_of_Pizza Integer
    0 <= Number_of_Popcorn Integer
    0 <= Number_of_Popsicle Integer
    0 <= Number_of_Soda Integer



**In [4]:**

```python
prob.solve()

# Is the solution optimal?
print("Status:", LpStatus[prob.status])
# Each of the variables is printed with it's value
for v in prob.variables():
    print(v.name, "=", v.varValue)
# The optimised objective function value is printed to the screen    
print("Minimum Number of Items = ", value(prob.objective))
```

    Status: Optimal
    
    Number_of_Beer = 0.0
    Number_of_Bottled_Water = 0.0
    Number_of_Chocolate_Bar = 0.0
    Number_of_Chocolate_Dipped_Cone = 0.0
    Number_of_Gummy_Bears = 0.0
    Number_of_Hamburger = 0.0
    Number_of_Hot_Dog = 0.0
    Number_of_Ice_Cream_Sandwich = 0.0
    Number_of_Licorice_Rope = 0.0
    Number_of_Nachos = 0.0
    Number_of_Pizza = 5.0
    Number_of_Popcorn = 0.0
    Number_of_Popsicle = 0.0
    Number_of_Soda = 0.0
    
    Minimum Number of Items =  5.0
