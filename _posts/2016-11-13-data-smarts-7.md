---
layout: post
title: "Data smarts in python- 3.0 naive Bayes"
tags:
    - python
    - notebook
---
Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch03.zip, Mandrill.xlsx

**In [1]:**

```python
# code written in py_3.0

import pandas as pd
import numpy as np
```

Load HAM training data - i.e., tweets about the product

**In [2]:**

```python
# find path to your Mandrill.xlsx
df_ham = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch03/Mandrill.xlsx','rb'), sheetname=0)
df_ham = df_ham.iloc[0:, 0:1]
df_ham.head() # use .head() to just show top 5 results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[blog] Using Nullmailer and Mandrill for your ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[blog] Using Postfix and free Mandrill email s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@aalbertson There are several reasons emails g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@adrienneleigh I just switched it over to Mand...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@ankeshk +1 to @mailchimp We use MailChimp for...</td>
    </tr>
  </tbody>
</table>
</div>

<!--more-->

Load SPAM training data - i.e., tweets **NOT** about the product

**In [3]:**

```python
df_spam = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch03/Mandrill.xlsx','rb'), sheetname=1)
df_spam = df_spam.iloc[0:, 0:1]
df_spam.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>¿En donde esta su remontada Mandrill?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>.@Katie_PhD Alternate, 'reproachful mandrill' ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>.@theophani can i get "drill" in there? it wou...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>“@ChrisJBoyland: Baby Mandrill Paignton Zoo 29...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“@MISSMYA #NameAnAmazingBand MANDRILL!” Mint C...</td>
    </tr>
  </tbody>
</table>
</div>



Install Natural Language Toolkit: http://www.nltk.org/install.html. You may also need to download nltk's dictionaries

**In [4]:**

```python
# python -m nltk.downloader punkt
```

**In [5]:**

```python
from nltk.tokenize import word_tokenize

test = df_ham.Tweet[0]
print(word_tokenize(test))
```

    ['[', 'blog', ']', 'Using', 'Nullmailer', 'and', 'Mandrill', 'for', 'your', 'Ubuntu', 'Linux', 'server', 'outboud', 'mail', ':', 'http', ':', '//bit.ly/ZjHOk7', '#', 'plone']


Following Marco Bonzanini's example https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/, I setup a pre-processing chain that recognises '@-mentions', 'emoticons', 'URLs' and '#hash-tags' as tokens

**In [6]:**

```python
import re

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
```

**In [7]:**

```python
print(preprocess(test))
{% endhighlight %}

    ['[', 'blog', ']', 'Using', 'Nullmailer', 'and', 'Mandrill', 'for', 'your', 'Ubuntu', 'Linux', 'server', 'outboud', 'mail', ':', 'http://bit.ly/ZjHOk7', '#plone']


**In [8]:**

```python
tweet = preprocess(test)
tweet
```




    ['[',
     'blog',
     ']',
     'Using',
     'Nullmailer',
     'and',
     'Mandrill',
     'for',
     'your',
     'Ubuntu',
     'Linux',
     'server',
     'outboud',
     'mail',
     ':',
     'http://bit.ly/ZjHOk7',
     '#plone']



Remove common stop-words + the non-default stop-words: 'RT' (i.e., re-tweet), 'via' (used in mentions), and ellipsis '…'

**In [9]:**

```python
from nltk.corpus import stopwords
import string

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '…', '¿', '“', '”']
```

**In [10]:**

```python
tweet_stop = [term for term in preprocess(test) if term not in stop]
```

**In [11]:**

```python
tweet_stop
```




    ['blog',
     'Using',
     'Nullmailer',
     'Mandrill',
     'Ubuntu',
     'Linux',
     'server',
     'outboud',
     'mail',
     'http://bit.ly/ZjHOk7',
     '#plone']



**In [12]:**

```python
from collections import Counter

count_all = Counter()
for tweet in df_ham.Tweet:
    # Create a list with all the terms
    terms_all = [term for term in preprocess(tweet) if term not in stop]
    # Update the counter
    count_all.update(terms_all)
# Print the first 5 most frequent words
print(count_all.most_common(10))
```

    [('Mandrill', 86), ('http://help.mandrill.com', 22), ('email', 21), ('I', 18), ('request', 16), ('@mandrillapp', 14), ('details', 13), ('emails', 13), ('de', 12), ('mandrill', 12)]


start

**In [13]:**

```python
df_ham["Tweet"] = df_ham["Tweet"].str.lower()

clean = []
for row in df_ham["Tweet"]:
    tweet = [term for term in preprocess(row) if term not in stop]
    clean.append(' '.join(tweet))

df_ham["Tweet"] = clean  # we now have clean tweets
df_ham["Class"] = 'ham' # add classification
df_ham.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>blog using nullmailer mandrill ubuntu linux se...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blog using postfix free mandrill email service...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@aalbertson several reasons emails go spam min...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@adrienneleigh switched mandrill let's see imp...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@ankeshk 1 @mailchimp use mailchimp marketing ...</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>



**In [14]:**

```python
df_spam["Tweet"] = df_spam["Tweet"].str.lower()

clean = []
for row in df_spam["Tweet"]:
    tweet = [term for term in preprocess(row) if term not in stop]
    clean.append(' '.join(tweet))

df_spam["Tweet"] = clean  # we now have clean tweets
df_spam["Class"] = 'spam' # add classification
df_spam.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>en donde esta su remontada mandrill</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@katie_phd alternate reproachful mandrill cove...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@theophani get drill would picture mandrill ho...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@chrisjboyland baby mandrill paignton zoo 29 t...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@missmya #nameanamazingband mandrill mint cond...</td>
      <td>spam</td>
    </tr>
  </tbody>
</table>
</div>



**In [15]:**

```python
df_data = pd.concat([df_ham,df_spam])
df_data = df_data.reset_index(drop=True)
df_data = df_data.reindex(np.random.permutation(df_data.index))
df_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281</th>
      <td>spark mandrill theme #nerdatwork</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>188</th>
      <td>@mandrill n k n k n 5 k 4 correction</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>112</th>
      <td>mandrill webhooks interspire bounce processing...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>43</th>
      <td>@matt_pickett u want reach mailchimp mandrill ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>242</th>
      <td>gostei de um vídeo @youtube de @franciscodanrl...</td>
      <td>spam</td>
    </tr>
  </tbody>
</table>
</div>



**In [16]:**

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(df_data["Tweet"].values)
counts
```




    <300x1662 sparse matrix of type '<class 'numpy.int64'>'
    	with 3493 stored elements in Compressed Sparse Row format>



**In [17]:**

```python
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
targets = df_data['Class'].values
classifier.fit(counts, targets)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



testing data

**In [18]:**

```python
df_test = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch03/Mandrill.xlsx','rb'), sheetname=6)
df_test = df_test.iloc[0:, 2:3]
df_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Just love @mandrillapp transactional email ser...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@rossdeane Mind submitting a request at http:/...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@veroapp Any chance you'll be adding Mandrill ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@Elie__ @camj59 jparle de relai SMTP!1 million...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>would like to send emails for welcome, passwor...</td>
    </tr>
  </tbody>
</table>
</div>



**In [19]:**

```python
df_test["Tweet"] = df_test["Tweet"].str.lower()

clean = []
for row in df_test["Tweet"]:
    tweet = [term for term in preprocess(row) if term not in stop]
    clean.append(' '.join(tweet))

df_test["Tweet"] = clean  # we now have clean tweets
df_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love @mandrillapp transactional email service ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@rossdeane mind submitting request http://help...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@veroapp chance you'll adding mandrill support...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@elie__ @camj59 jparle de relai smtp 1 million...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>would like send emails welcome password resets...</td>
    </tr>
  </tbody>
</table>
</div>



Following Zac Stewart's example (see [here](http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html), and [here](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html)), we use sklearn's 'pipeline' feature to merge the feature extraction and classification into one operation

**In [20]:**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])

pipeline.fit(df_data['Tweet'].values, df_data['Class'].values)

df_test["Prediction Class"] = pipeline.predict(df_test['Tweet'].values) # add classification ['spam', 'ham']
df_test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Prediction Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love @mandrillapp transactional email service ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@rossdeane mind submitting request http://help...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@veroapp chance you'll adding mandrill support...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@elie__ @camj59 jparle de relai smtp 1 million...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>4</th>
      <td>would like send emails welcome password resets...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>5</th>
      <td>coworker using mandrill would entrust email ha...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@mandrill realised 5 seconds hitting send</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>7</th>
      <td>holy shit ’ http://www.mandrill.com/</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>8</th>
      <td>new subscriber profile page activity timeline ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@mandrillapp increases scalability http://bit....</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>10</th>
      <td>beets @missmya #nameanamazingband mandrill</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>11</th>
      <td>@luissand0val fernando vargas mandrill mexican...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>12</th>
      <td>photo oculi-ds mandrill natalie manuel http://...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>13</th>
      <td>@mandrill neither sadpanda together :(</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>14</th>
      <td>@mandrill n k n k n 5 k 4, long time think</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>15</th>
      <td>megaman x spark mandrill acapella http://youtu...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@angeluserrare1 storm eagle ftw nom ás dejes q...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gostei de um vídeo @youtube http://youtu.be/xz...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2 year-old mandrill jj thinking pic http://ow....</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>19</th>
      <td>120 years moscow zoo mandrill поста ссср #post...</td>
      <td>spam</td>
    </tr>
  </tbody>
</table>
</div>



**In [21]:**

```python
true_class = pd.read_excel(open('C:/Users/craigrshenton/Desktop/Dropbox/excel_data_sci/ch03/Mandrill.xlsx','rb'), sheetname=6)
df_test["True Class"] = true_class.iloc[0:, 1:2]
df_test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet</th>
      <th>Prediction Class</th>
      <th>True Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love @mandrillapp transactional email service ...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@rossdeane mind submitting request http://help...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@veroapp chance you'll adding mandrill support...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@elie__ @camj59 jparle de relai smtp 1 million...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>would like send emails welcome password resets...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>coworker using mandrill would entrust email ha...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@mandrill realised 5 seconds hitting send</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>7</th>
      <td>holy shit ’ http://www.mandrill.com/</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>8</th>
      <td>new subscriber profile page activity timeline ...</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@mandrillapp increases scalability http://bit....</td>
      <td>ham</td>
      <td>APP</td>
    </tr>
    <tr>
      <th>10</th>
      <td>beets @missmya #nameanamazingband mandrill</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>11</th>
      <td>@luissand0val fernando vargas mandrill mexican...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>12</th>
      <td>photo oculi-ds mandrill natalie manuel http://...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>13</th>
      <td>@mandrill neither sadpanda together :(</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>14</th>
      <td>@mandrill n k n k n 5 k 4, long time think</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>15</th>
      <td>megaman x spark mandrill acapella http://youtu...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@angeluserrare1 storm eagle ftw nom ás dejes q...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gostei de um vídeo @youtube http://youtu.be/xz...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2 year-old mandrill jj thinking pic http://ow....</td>
      <td>ham</td>
      <td>OTHER</td>
    </tr>
    <tr>
      <th>19</th>
      <td>120 years moscow zoo mandrill поста ссср #post...</td>
      <td>spam</td>
      <td>OTHER</td>
    </tr>
  </tbody>
</table>
</div>



Naturally, in a business application we will generally not have a set of independent test data available. To get around this, we can use *cross-validation*. Here, we split the training set into two parts, a large training set (~80%), and a smaller testing set (~20%). In this example we also repeat 6 times to average out the results using *k-fold cross-validation* and scikit-learn's 'KFold' function

**In [22]:**

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

print('Total emails classified:', len(df_data))
print('Score:', sum(scores)/len(scores))
```

    Total emails classified: 300
    Score: 0.836360280546


The F1 score is a measure of a test's accuracy, in both precision and recall. F1 score reaches its best value at 1 and worst at 0, so the model's score of 0.836 is not bad for a first pass

**In [23]:**

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
