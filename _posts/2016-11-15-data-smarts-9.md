---
layout: post
title: "Data smarts in python- 3.2 cross validation"
tags:
    - python
    - notebook
---
Continuing from chapter 3. Naturally, in a business application we will generally not have a set of independent test data available. To get around this, we can use *cross-validation*. Following Zac Stewart's example (see [here](http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html)), we split the training set into two parts, a large training set (~80%), and a smaller testing set (~20%). In this example we also repeat 6 times to average out the results using *k-fold cross-validation* and scikit-learn's 'KFold' function

**In [1]:**

```python
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

k_fold = KFold(n=len(df_data), n_folds=6)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = df_data.iloc[train_indices]['Tweet'].values
    train_y = df_data.iloc[train_indices]['Class'].values

    test_text = df_data.iloc[test_indices]['Tweet'].values
    test_y = df_data.iloc[test_indices]['Class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

print('Total tweets classified:', len(df_data))
print('Score:', sum(scores)/len(scores))
```

    Total tweets classified: 300
    Score: 0.836360280546


The F1 score is a measure of a test's accuracy, in both precision and recall. F1 score reaches its best value at 1 and worst at 0, so the model's score of 0.836 is not bad for a first pass given how noisy the tweet dataset is

<!--more-->

**In [2]:**

```python
print('Confusion matrix:')
print(confusion)
```

    Confusion matrix:
    [[144   6]
     [ 39 111]]


A confusion matrix helps us understand how the model performed for individual features. Out of the 300 tweets,
the model incorrectly classified about 39 tweets that are about the produt, and 6 tweets that are not

In order to improve the results there are two approaches we can take:
- We could improve the data pre-processing by cleaning the data with more filters
- We can tune the parameters of the naïve Bayes classifier
