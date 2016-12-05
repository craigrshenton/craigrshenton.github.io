---
layout: post
title: "notebook6"
tags:
    - python
    - notebook
---
Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch06.zip, RetailMart.xlsx

**In [1]:**

{% highlight python %}
# code written in py_3.0

import pandas as pd
import numpy as np
{% endhighlight %}

Load customer account data - i.e., past product sales data

**In [2]:**

{% highlight python %}
# find path to your RetailMart.xlsx
df_accounts = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch06/RetailMart.xlsx','rb'), sheetname=0)
df_accounts = df_accounts.drop('Unnamed: 17', 1) # drop empty col
df_accounts.rename(columns={'PREGNANT':'Pregnant'}, inplace=True)
df_accounts.rename(columns={'Home/Apt/ PO Box':'Residency'}, inplace=True) # add simpler col name
df_accounts.columns = [x.strip().replace(' ', '_') for x in df_accounts.columns] # python does not like spaces in var names
df_accounts.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Implied_Gender</th>
      <th>Residency</th>
      <th>Pregnancy_Test</th>
      <th>Birth_Control</th>
      <th>Feminine_Hygiene</th>
      <th>Folic_Acid</th>
      <th>Prenatal_Vitamins</th>
      <th>Prenatal_Yoga</th>
      <th>Body_Pillow</th>
      <th>Ginger_Ale</th>
      <th>Sea_Bands</th>
      <th>Stopped_buying_ciggies</th>
      <th>Cigarettes</th>
      <th>Smoking_Cessation</th>
      <th>Stopped_buying_wine</th>
      <th>Wine</th>
      <th>Maternity_Clothes</th>
      <th>Pregnant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>H</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U</td>
      <td>H</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We need to categorise the 'Pregnant' column so that it can only take on one of two (in this case) possabilities. Here 1 = pregnant, and 0 = not pregnant

**In [3]:**

{% highlight python %}
df_accounts['Pregnant'] = df_accounts['Pregnant'].astype('category') # set col type
{% endhighlight %}

Following Greg Lamp over at the Yhat Blog (see [here](http://blog.yhat.com/posts/logistic-regression-python-rodeo.html)), we need to 'dummify' (i.e., separate out) the catagorical variables: gender and residency

**In [4]:**

{% highlight python %}
# dummify gender var
dummy_gender = pd.get_dummies(df_accounts['Implied_Gender'], prefix='Gender')
print(dummy_gender.head())
{% endhighlight %}

       Gender_F  Gender_M  Gender_U
    0         0         1         0
    1         0         1         0
    2         0         1         0
    3         0         0         1
    4         1         0         0
    

**In [5]:**

{% highlight python %}
# dummify residency var
dummy_resident = pd.get_dummies(df_accounts['Residency'], prefix='Resident')
print(dummy_resident.head())
{% endhighlight %}

       Resident_A  Resident_H  Resident_P
    0           1           0           0
    1           0           1           0
    2           0           1           0
    3           0           1           0
    4           1           0           0
    

**In [6]:**

{% highlight python %}
# make clean dataframe for regression model
cols_to_keep = df_accounts.columns[2:len(df_accounts.columns)-1] # keep all but 'Pregnant' var
# add dummy vars back in
data = pd.concat([dummy_gender.ix[:, 'Gender_M':],dummy_resident.ix[:, 'Resident_H':],df_accounts[cols_to_keep]], axis=1)
data.insert(0, 'Intercept', 1.0) # manually add the intercept
data.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>Gender_M</th>
      <th>Gender_U</th>
      <th>Resident_H</th>
      <th>Resident_P</th>
      <th>Pregnancy_Test</th>
      <th>Birth_Control</th>
      <th>Feminine_Hygiene</th>
      <th>Folic_Acid</th>
      <th>Prenatal_Vitamins</th>
      <th>Prenatal_Yoga</th>
      <th>Body_Pillow</th>
      <th>Ginger_Ale</th>
      <th>Sea_Bands</th>
      <th>Stopped_buying_ciggies</th>
      <th>Cigarettes</th>
      <th>Smoking_Cessation</th>
      <th>Stopped_buying_wine</th>
      <th>Wine</th>
      <th>Maternity_Clothes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <th>2</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**In [7]:**

{% highlight python %}
from patsy import dmatrices
import statsmodels.api as sm

train_cols = data.columns[1:]
logit = sm.Logit(df_accounts['Pregnant'], data[train_cols])

# fit the model
result = logit.fit()
{% endhighlight %}

    Optimization terminated successfully.
             Current function value: 0.373878
             Iterations 8
    

**In [8]:**

{% highlight python %}
print('Parameters:')
print(result.params)
print(result.summary())
{% endhighlight %}

    Parameters:
    Gender_M                 -0.616638
    Gender_U                 -0.021343
    Resident_H               -0.367855
    Resident_P               -0.209655
    Pregnancy_Test            2.319395
    Birth_Control            -2.400243
    Feminine_Hygiene         -2.084057
    Folic_Acid                4.048098
    Prenatal_Vitamins         2.402392
    Prenatal_Yoga             2.969468
    Body_Pillow               1.256653
    Ginger_Ale                1.884790
    Sea_Bands                 0.940477
    Stopped_buying_ciggies    1.218470
    Cigarettes               -1.528444
    Smoking_Cessation         1.750957
    Stopped_buying_wine       1.276292
    Wine                     -1.676815
    Maternity_Clothes         2.003167
    dtype: float64
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               Pregnant   No. Observations:                 1000
    Model:                          Logit   Df Residuals:                      981
    Method:                           MLE   Df Model:                           18
    Date:                Mon, 05 Dec 2016   Pseudo R-squ.:                  0.4606
    Time:                        00:46:36   Log-Likelihood:                -373.88
    converged:                       True   LL-Null:                       -693.15
                                            LLR p-value:                6.050e-124
    ==========================================================================================
                                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------------------
    Gender_M                  -0.6166      0.177     -3.481      0.000        -0.964    -0.269
    Gender_U                  -0.0213      0.295     -0.072      0.942        -0.599     0.556
    Resident_H                -0.3679      0.165     -2.228      0.026        -0.691    -0.044
    Resident_P                -0.2097      0.317     -0.662      0.508        -0.830     0.411
    Pregnancy_Test             2.3194      0.525      4.418      0.000         1.290     3.348
    Birth_Control             -2.4002      0.360     -6.670      0.000        -3.106    -1.695
    Feminine_Hygiene          -2.0841      0.337     -6.182      0.000        -2.745    -1.423
    Folic_Acid                 4.0481      0.764      5.300      0.000         2.551     5.545
    Prenatal_Vitamins          2.4024      0.372      6.463      0.000         1.674     3.131
    Prenatal_Yoga              2.9695      1.158      2.565      0.010         0.701     5.238
    Body_Pillow                1.2567      0.860      1.462      0.144        -0.429     2.942
    Ginger_Ale                 1.8848      0.428      4.401      0.000         1.045     2.724
    Sea_Bands                  0.9405      0.671      1.401      0.161        -0.376     2.256
    Stopped_buying_ciggies     1.2185      0.340      3.586      0.000         0.552     1.885
    Cigarettes                -1.5284      0.365     -4.185      0.000        -2.244    -0.813
    Smoking_Cessation          1.7510      0.514      3.405      0.001         0.743     2.759
    Stopped_buying_wine        1.2763      0.302      4.232      0.000         0.685     1.867
    Wine                      -1.6768      0.340     -4.927      0.000        -2.344    -1.010
    Maternity_Clothes          2.0032      0.330      6.074      0.000         1.357     2.650
    ==========================================================================================
    

logistic reg revisited with sklearn

**In [9]:**

{% highlight python %}
# define X and y
X = data[train_cols]
y = df_accounts['Pregnant']

# train/test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# make predictions for testing set
y_pred_class = logreg.predict(X_test)

# calculate testing accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
{% endhighlight %}

    0.88
    

    D:\Anaconda\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

**In [21]:**

{% highlight python %}
# predict probability of survival
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt

# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
{% endhighlight %}


![png]({{ site.baseurl }}/notebooks/notebook6_files/notebook6_14_0.png)


**In [11]:**

{% highlight python %}
# calculate AUC
print(metrics.roc_auc_score(y_test, y_pred_prob))
{% endhighlight %}

    0.94394259722
    

**In [12]:**

{% highlight python %}
# histogram of predicted probabilities grouped by actual response value
df = pd.DataFrame({'probability':y_pred_prob, 'actual':y_test})
df.hist(column='probability', by='actual', sharex=True, sharey=True)
plt.show()
{% endhighlight %}


![png]({{ site.baseurl }}/notebooks/notebook6_files/notebook6_16_0.png)


**In [13]:**

{% highlight python %}
# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
{% endhighlight %}




    0.89871999999999996



Random forest feature selection

**In [211]:**

{% highlight python %}
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
clf.fit(data[train_cols], df_accounts['Pregnant'])
{% endhighlight %}




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=500, n_jobs=-1, oob_score=False,
                random_state=None, verbose=0, warm_start=False)



**In [212]:**

{% highlight python %}
# sort the features by importance
sorted_idx = clf.feature_importances_
df_features = pd.DataFrame({"Feature": train_cols})
df_features['Importance'] = sorted_idx

df_features = df_features.sort_values(by=['Importance'], ascending=[True]) # sort my most important feature
ax = df_features.plot(kind='barh', title ="Classification Feature Importance", figsize=(15, 10), legend=False, fontsize=12)
ax.set_xlabel("Importance", fontsize=12)
ax.set_yticklabels(df_features['Feature'])
plt.show()
{% endhighlight %}


![png]({{ site.baseurl }}/notebooks/notebook6_files/notebook6_20_0.png)


We can see that the purchase of Folic Acid is a much better predictor of a customer pregnancy, surprisingly more so than an intrest in Prenatal Yoga (presumably more expectant mother use folic acid than take up yoga)---this information could be used to accurately target the advertisment of baby products

**In [None]:**

{% highlight python %}

{% endhighlight %}
