---
layout: post
title: "Data smarts in python- 3.0 naive Bayes"
tags:
    - python
    - notebook
---
Continuing from chapter 3. Load data from http://media.wiley.com/product_ancillary/6X/11186614/DOWNLOAD/ch03.zip, Mandrill.xlsx

**In [1]:**

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
    <tr>
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

<!--more-->

**In [2]:**

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
    <tr>
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

Add both dataset together and randomise

**In [3]:**

```python
df_data = pd.concat([df_ham,df_spam])
df_data = df_data.reset_index(drop=True)
df_data = df_data.reindex(np.random.permutation(df_data.index))
df_data.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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


**In [3]:**

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(df_data["Tweet"].values)
counts
```




    <300x1662 sparse matrix of type '<class 'numpy.int64'>'
    	with 3493 stored elements in Compressed Sparse Row format>



**In [4]:**

```python
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
targets = df_data['Class'].values
classifier.fit(counts, targets)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



Load test data

**In [5]:**

```python
df_test = pd.read_excel(open('.../Mandrill.xlsx','rb'), sheetname=6)
df_test = df_test.iloc[0:, 2:3]
df_test.head()
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

Preprocess test data

**In [6]:**

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
    <tr>
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

**In [7]:**

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
    <tr>
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

Compare model classification predictions against known classifications

**In [8]:**

```python
true_class = pd.read_excel(open('.../Mandrill.xlsx','rb'), sheetname=6)
df_test["True Class"] = true_class.iloc[0:, 1:2]
df_test
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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

We can see that out of 20 test tweets the model correctly predicts 19 tweet classifications