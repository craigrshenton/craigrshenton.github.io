---
layout: post
title: "Data smarts in python- 3.0 text preprocessing"
tags:
    - python
    - notebook
---

Contunuing with my python translation of John W. Foreman's 2014 book "Data Smart: Using Data Science to Transform Information into Insight". Chapter 3 covers a naive Bayes model for text classification. 
In this first section we look at preprocessing twitter data using the python natural language toolkit 'nltk'

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
df_ham = pd.read_excel(open('.../Mandrill.xlsx','rb'), sheetname=0)
df_ham = df_ham.iloc[0:, 0:1]
df_ham.head() # use .head() to just show top 5 results
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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
df_spam = pd.read_excel(open('.../Mandrill.xlsx','rb'), sheetname=1)
df_spam = df_spam.iloc[0:, 0:1]
df_spam.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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


Install Natural Language Toolkit: http://www.nltk.org/install.html. You may also need to download nltk's dictionaries using > python -m nltk.downloader punkt (in terminal)


**In [4]:**

```python
from nltk.tokenize import word_tokenize

test = df_ham.Tweet[0]
print(word_tokenize(test))
```

    ['[', 'blog', ']', 'Using', 'Nullmailer', 'and', 'Mandrill', 'for', 'your', 'Ubuntu', 'Linux', 'server', 'outboud', 'mail', ':', 'http', ':', '//bit.ly/ZjHOk7', '#', 'plone']


Following Marco Bonzanini's example https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/, we setup a pre-processing chain that recognises '@-mentions', 'emoticons', 'URLs' and '#hash-tags' as tokens

**In [5]:**

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

**In [6]:**

```python
print(preprocess(test))
```

    ['[', 'blog', ']', 'Using', 'Nullmailer', 'and', 'Mandrill', 'for', 'your', 'Ubuntu', 'Linux', 'server', 'outboud', 'mail', ':', 'http://bit.ly/ZjHOk7', '#plone']


**In [7]:**

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

**In [8]:**

```python
from nltk.corpus import stopwords
import string

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '…', '¿', '“', '”']
```

**In [9]:**

```python
tweet_stop = [term for term in preprocess(test) if term not in stop]
```

**In [10]:**

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


Now we can apply the preprocessing to all tweets and count which are most often used when talking about the product

**In [11]:**

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



