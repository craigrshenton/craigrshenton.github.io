---
layout: post
title: "Data smarts in python- 1.1 bar chart"
tags:
    - ipython
    - notebook
---

Continuing from chapter 1. Plotting a basic bar chart in python.

**In [1]:**

```python
df_cals = df_cals.set_index('Item') # index df by items
# Items ranked by calories = .sort_values(by='Calories',ascending=True)
# rot = axis rotation
ax = df_cals.sort_values(by='Calories',ascending=True).plot(kind='bar', title ="Calories",figsize=(15,5),legend=False, fontsize=10, alpha=0.75, rot=20,)
plt.xlabel("") # no x-axis lable
plt.show()
```


![png]({{ site.baseurl }}/notebooks/notebook_files/notebook_9_0.png)
