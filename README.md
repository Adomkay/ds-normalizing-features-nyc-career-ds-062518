
# Normalizing Features

You might have noticed that our previous model wasn't exactly stellar. This is because we missed a key technique used in machine learning: normalization. Normalizing features takes all of our data and fits variables to a similar scale and range. When we previously performed gradient descent, the budget and imdbVotes features had a much larger impact on our steps and the resulting output. That's not necessarily due to those features being more predictive of the gross domestic sales, but rather simply because the values of those features were much higher then the imdbRating or Metascore features which had much more narrow ranges. To account for this, we'll start normalizing our data, and transform back to the raw version when needed.



```python
import pandas as pd
%matplotlib inline
```


```python
df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>Response_Json</th>
      <th>Year</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>0</td>
      <td>2008</td>
      <td>6.8</td>
      <td>48</td>
      <td>206513</td>
      <td>4.912759e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0</td>
      <td>2012</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.267265e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>0</td>
      <td>2013</td>
      <td>8.1</td>
      <td>96</td>
      <td>537525</td>
      <td>1.626624e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>0</td>
      <td>2013</td>
      <td>6.7</td>
      <td>55</td>
      <td>173726</td>
      <td>7.723381e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
      <td>0</td>
      <td>2013</td>
      <td>7.5</td>
      <td>62</td>
      <td>74170</td>
      <td>4.151958e+07</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Basic Norm function
Write a function norm(col) that takes in a pandas series, and rescales the data to have a minimum of zero and a maximum of 1. Think about how you can do this by simply using the minimum and maximum of the column.


```python
def norm(col):
    minimum = col.min()
    maximum = col.max()
    return (col-maximum)/(maximum-minimum)
```

### 2. Apply your norm function to the X feature columns


```python
cols = ['budget',  'imdbRating', 'Metascore', 'imdbVotes']
for col in cols:
    df[col] = norm(df[col])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>Response_Json</th>
      <th>Year</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.965831</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>0</td>
      <td>2008</td>
      <td>-0.160494</td>
      <td>-0.500000</td>
      <td>-0.615808</td>
      <td>4.912759e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.817044</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0</td>
      <td>2012</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>2.267265e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.933941</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>0</td>
      <td>2013</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.626624e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.747153</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>0</td>
      <td>2013</td>
      <td>-0.172840</td>
      <td>-0.427083</td>
      <td>-0.676804</td>
      <td>7.723381e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.842825</td>
      <td>95020213</td>
      <td>42</td>
      <td>0</td>
      <td>2013</td>
      <td>-0.074074</td>
      <td>-0.354167</td>
      <td>-0.862016</td>
      <td>4.151958e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df[['budget', 'imdbRating',
       'Metascore', 'imdbVotes']]
y = df['domgross']
```

### 3. Try writing a slightly different normalization function: the mean normaliztion.
Here's how its defined:  
mean_normalized_x = $\frac{x-mean(x)}{max(x)-min(x)}$


```python
def norm(col):
    minimum = col.min()
    maximum = col.max()
    return (col-np.mean(col))/(maximum-minimum)
```
