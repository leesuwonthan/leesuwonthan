# "Introduction"

> "This project is part of my coursework in the university.
Here, I performed sentiment analysis by using a pre-trained model to predict the sentiment that indicate whether the customer being satisfied or not from the service. I will be working with a data set, Amazon Fine Food Reviews, which provide attribute such as review comment, summary, score, profilename etc. With that, I might choose only some atrribute and split them into training set and testing set. In the end, I would show the accuracy rate of my model and some challenge for develop the efficientcy of model in the future."

- toc: true
- badges: true
- comments: true
- categories: [python]



```python
#import libraries
#the dataset: Amazon Fine Food Reviews
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime as dt
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
```

# Import the data set 


```python
#import dataset
df = pd.read_csv('Reviews.csv')
```

```python
#see the sample of dataset
df.sample(100000)
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
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>392858</th>
      <td>392859</td>
      <td>B003VXL0V6</td>
      <td>A9NTRP272B3HT</td>
      <td>Judy</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1343692800</td>
      <td>Close to expiration date.</td>
      <td>Coffee quality is as good as expected but, clo...</td>
    </tr>
    <tr>
      <th>141490</th>
      <td>141491</td>
      <td>B0047RQ9M0</td>
      <td>A270KH15QHV8SY</td>
      <td>Richard Rudeen</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>1302393600</td>
      <td>Great coffee</td>
      <td>I had received a Keurig coffee maker for the h...</td>
    </tr>
    <tr>
      <th>418590</th>
      <td>418591</td>
      <td>B001F0RRU0</td>
      <td>A17S5C0YLCREZY</td>
      <td>Random passerby #3147</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>1279152000</td>
      <td>Great treats, but...</td>
      <td>They are a bit too easy for my dog to get them...</td>
    </tr>
    <tr>
      <th>146505</th>
      <td>146506</td>
      <td>B000WFKWDI</td>
      <td>A38VKLWPITZ8XS</td>
      <td>Dallas R. Smith "Mittens"</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>1328659200</td>
      <td>Kitty Loves This</td>
      <td>My 14 year old cat is in chronic renal failure...</td>
    </tr>
    <tr>
      <th>192560</th>
      <td>192561</td>
      <td>B006GA666U</td>
      <td>ACXXJ9S2112YY</td>
      <td>S. Wilson "Word Girl"</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1268006400</td>
      <td>Underwhelming Brew</td>
      <td>I found this coffee to have a robust flavor, b...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>536562</th>
      <td>536563</td>
      <td>B001EO6GU4</td>
      <td>A9EHQGPXDBDJK</td>
      <td>Patty P.</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1329091200</td>
      <td>My dog LOVES these treats</td>
      <td>We had to take our dog off regular treats due ...</td>
    </tr>
    <tr>
      <th>122995</th>
      <td>122996</td>
      <td>B004X8DO8U</td>
      <td>A2972NS4FP8TV</td>
      <td>reemas "holla"</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1327363200</td>
      <td>Not bad but not quite there</td>
      <td>These pickles are spicy but to truly enjoy the...</td>
    </tr>
    <tr>
      <th>393888</th>
      <td>393889</td>
      <td>B003L4BM3G</td>
      <td>A356V3JU6LGGBT</td>
      <td>J. M. Jackson "Smilely J"</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1293580800</td>
      <td>So good</td>
      <td>Used my grandmothers old hot air popper!  It d...</td>
    </tr>
    <tr>
      <th>545837</th>
      <td>545838</td>
      <td>B0026LFFN8</td>
      <td>A38SPKGCW19ZIF</td>
      <td>M. Blalock "L. Blalock"</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1307318400</td>
      <td>She tolerates this food well!</td>
      <td>I have a 5 yr old cat that has a sensitive sto...</td>
    </tr>
    <tr>
      <th>146495</th>
      <td>146496</td>
      <td>B000WFKWDI</td>
      <td>A8P8KPVXCWV9R</td>
      <td>Bryan J. Kautzman "BK"</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>1328659200</td>
      <td>The best cat food on the planet</td>
      <td>I have a seven year old gray tabby named Buddy...</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 10 columns</p>
</div>



# Reshape and Explore Data

Initially, we need to explore the landscape of our data first and make a decision to selecte only essential attributes. Also, we can perform visualization in order to understand our data more.


```python
#get the column name using list comprehension
print([col for col in df])  
```

    ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']
    


```python
#dataset is too large and it requires more resources to utilize all dataset
#dislaimer: in this project, I will used only one-fifth of the dataset
df.shape 
```




    (568454, 10)




```python
df0 = df.sample(frac = 0.20) # taking 20% of dataset
df0 = df0[['Id','ProfileName','Score', 'Time', 'Summary', 'Text']] # query only some attribute
df0.head()
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
      <th>Id</th>
      <th>ProfileName</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>396384</th>
      <td>396385</td>
      <td>AMBER</td>
      <td>5</td>
      <td>1347494400</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I...</td>
    </tr>
    <tr>
      <th>10280</th>
      <td>10281</td>
      <td>Connie Moreno</td>
      <td>3</td>
      <td>1257292800</td>
      <td>Convenient but disappointing</td>
      <td>These taste just like any other packaged pork ...</td>
    </tr>
    <tr>
      <th>348099</th>
      <td>348100</td>
      <td>M. Abendroth</td>
      <td>5</td>
      <td>1328659200</td>
      <td>One of the best</td>
      <td>For those who don't know, cacao nibs are small...</td>
    </tr>
    <tr>
      <th>550020</th>
      <td>550021</td>
      <td>Maggie Leave "Maggie H"</td>
      <td>5</td>
      <td>1332028800</td>
      <td>This is the best tea I've got in the U.S.</td>
      <td>Much better than other teas I've tried. Real t...</td>
    </tr>
    <tr>
      <th>142452</th>
      <td>142453</td>
      <td>mom 94 "carin"</td>
      <td>4</td>
      <td>1228521600</td>
      <td>very dense finished product</td>
      <td>I have purchased this item 3 times and have ha...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create an id of dataset (more organized)
id = np.arange(0,df0.shape[0]) 
id.shape
```




    (113691,)




```python
df0['id'] = id # insert new_id that has been created
df0.set_index("id", inplace = True) #setting as index_column
df0.pop('Id') # taking out the old one
df0
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
      <th>ProfileName</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AMBER</td>
      <td>5</td>
      <td>1347494400</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Connie Moreno</td>
      <td>3</td>
      <td>1257292800</td>
      <td>Convenient but disappointing</td>
      <td>These taste just like any other packaged pork ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M. Abendroth</td>
      <td>5</td>
      <td>1328659200</td>
      <td>One of the best</td>
      <td>For those who don't know, cacao nibs are small...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Maggie Leave "Maggie H"</td>
      <td>5</td>
      <td>1332028800</td>
      <td>This is the best tea I've got in the U.S.</td>
      <td>Much better than other teas I've tried. Real t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mom 94 "carin"</td>
      <td>4</td>
      <td>1228521600</td>
      <td>very dense finished product</td>
      <td>I have purchased this item 3 times and have ha...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113686</th>
      <td>Cheryl L. Malone</td>
      <td>5</td>
      <td>1337817600</td>
      <td>Very Nice</td>
      <td>This seems to be a very good product.  What my...</td>
    </tr>
    <tr>
      <th>113687</th>
      <td>Lisa Gump "Home CEO"</td>
      <td>5</td>
      <td>1137542400</td>
      <td>Wonderful afternoon treat!</td>
      <td>I did not buy this from this particular vendor...</td>
    </tr>
    <tr>
      <th>113688</th>
      <td>T</td>
      <td>5</td>
      <td>1327190400</td>
      <td>Great Food for Multi-Cat Household Also!</td>
      <td>I have seven cats and all of them do well with...</td>
    </tr>
    <tr>
      <th>113689</th>
      <td>markw</td>
      <td>5</td>
      <td>1325808000</td>
      <td>Gorgeous!</td>
      <td>I've never seen another schilleriana with such...</td>
    </tr>
    <tr>
      <th>113690</th>
      <td>A. Damuth "build stuff!"</td>
      <td>4</td>
      <td>1301702400</td>
      <td>Works well for our bread recipe.</td>
      <td>Our favorite yeast used to be some sold at the...</td>
    </tr>
  </tbody>
</table>
<p>113691 rows × 5 columns</p>
</div>




```python
#rearrange the position of atributes (to be more organized)
df1 = df0[['Time', 'ProfileName', 'Summary', 'Text', 'Score']] 
df1.head(20) 
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
      <th>Time</th>
      <th>ProfileName</th>
      <th>Summary</th>
      <th>Text</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1347494400</td>
      <td>AMBER</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1257292800</td>
      <td>Connie Moreno</td>
      <td>Convenient but disappointing</td>
      <td>These taste just like any other packaged pork ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1328659200</td>
      <td>M. Abendroth</td>
      <td>One of the best</td>
      <td>For those who don't know, cacao nibs are small...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1332028800</td>
      <td>Maggie Leave "Maggie H"</td>
      <td>This is the best tea I've got in the U.S.</td>
      <td>Much better than other teas I've tried. Real t...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1228521600</td>
      <td>mom 94 "carin"</td>
      <td>very dense finished product</td>
      <td>I have purchased this item 3 times and have ha...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1323993600</td>
      <td>Judy</td>
      <td>Good for the money</td>
      <td>The cashews were nice and big, with the vast m...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1296604800</td>
      <td>Chicker</td>
      <td>Not very good in comparison</td>
      <td>It tasted like fake bananas.  I gave it a try ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1326585600</td>
      <td>Zack Davisson</td>
      <td>Even if you like curry ...</td>
      <td>This is apparently falling into that "love it ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1290816000</td>
      <td>S. Fullerton</td>
      <td>How can this be called a "gift" pack?</td>
      <td>I got this as a Christmas gift and am now emba...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1162512000</td>
      <td>John Williams</td>
      <td>Beetlejuice - Tim Burton's Masterpiece</td>
      <td>When the world didn't know who Tim Burton was,...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1213401600</td>
      <td>dog lover</td>
      <td>Great dog treats</td>
      <td>My 3 maltese love these.  The product is great...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1338422400</td>
      <td>K. Pollock</td>
      <td>Baby Gourmet Organic Purees</td>
      <td>My grandson loves baby gourmet! My son and dau...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1278892800</td>
      <td>mispolomino13</td>
      <td>False Advertising</td>
      <td>I paid a much higher price for these plants, t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1334016000</td>
      <td>APC "apc"</td>
      <td>This is the equivalent of kitty crack!</td>
      <td>My cats go INSANE for these treats. I am a vet...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1350604800</td>
      <td>NotLime</td>
      <td>Sage Tea helps with night sweats</td>
      <td>I have one cup a day and it really decreases m...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1299715200</td>
      <td>katyliz2</td>
      <td>Awesome Apricot &amp; Almond!</td>
      <td>These are truly delicious, full of nuts, and t...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1326844800</td>
      <td>Jewvonne</td>
      <td>Excellent tasting pears</td>
      <td>Libby's makes the best canned pear halves with...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1332201600</td>
      <td>Tyke</td>
      <td>splendid splenda</td>
      <td>So much easier than carrying around the indivi...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1280188800</td>
      <td>Bliss 149</td>
      <td>It's Actually GOOD</td>
      <td>This is the only protein I truly enjoy. I set ...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1327536000</td>
      <td>Jason Sepe</td>
      <td>Delicious coffee!</td>
      <td>This is great coffee! I have a Bialetti Moka c...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape
```




    (113691, 5)




```python
#Explore the dataset
#Goal: visualize the proportion of reviews catagrorized by score
```


```python
#showing proportion of each score rate in percentage
score_prop = df1.groupby('Score')['Text'].count()/len(df1.Score)*100
round(score_prop)
```




    Score
    1     9.0
    2     5.0
    3     8.0
    4    14.0
    5    64.0
    Name: Text, dtype: float64




```python
#Visualize proportion of score with pie chart
# declaring data
x = score_prop.to_list()
data = x
keys = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
  
# define Seaborn color palette to use
palette_color = sns.color_palette('RdBu')
  
# plotting data on chart
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
  
# displaying chart
plt.show()

#NOTE: The marjority of the plot is dominated by the reviews with Score 5, and this could lead to imbalance of data prediction.
```


    
![png](output_16_0.png)
    



```python
# Explore the data
# displaying the full text of reviews
with pd.option_context('display.max_colwidth', None):
  display(df1)

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
      <th>Time</th>
      <th>ProfileName</th>
      <th>Summary</th>
      <th>Text</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1347494400</td>
      <td>AMBER</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I already use daily. I am totally happy with this new Splenda product. The biggest improvement I could come up with is it doesn't like to mix in a few liquids and a bigger bag would always be welcome.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1257292800</td>
      <td>Connie Moreno</td>
      <td>Convenient but disappointing</td>
      <td>These taste just like any other packaged pork rind product I've tried. I was hoping these would be fresher, closer to the real thing I buy in Mexican markets. Oh well, for a quick snack when you don't feel like leaving the house, it's fine.</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1328659200</td>
      <td>M. Abendroth</td>
      <td>One of the best</td>
      <td>For those who don't know, cacao nibs are small pieces of the cacao beans from which chocolate is made.  They are roasted much like coffee beans, and even have a somewhat similar nutty flavor, in addition to being chocolatey.&lt;br /&gt;&lt;br /&gt;If you like dark chocolate and you like it crunchy with nibs, this is one of the best.  It's like the gourmet version of a Nestle Crunch bar.  The nibs are evenly distributed at a density level that adds the perfect crunch without being overbearing, and provides a snap of darkness that contrasts somewhat with the sweetness of the chocolate in which they are embedded.  My other favorite from Endangered Species is the 88% dark panther bar.  This one is a little sweeter than that, since it's only 72% cacao, so it should be more palatable to people who find that level of darkness to be a bit too much.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1332028800</td>
      <td>Maggie Leave "Maggie H"</td>
      <td>This is the best tea I've got in the U.S.</td>
      <td>Much better than other teas I've tried. Real tea lovers would drink loose leaf tea instead of tea bags, and this tea is definitely the best that I've ever got.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1228521600</td>
      <td>mom 94 "carin"</td>
      <td>very dense finished product</td>
      <td>I have purchased this item 3 times and have had positive results each time.  Though the finished product is very dense and not as sweet as the traditional Jiffy Corn Muffin, if you have gluten intolerance, you will find this to be an adequate replacement.  It does offer lactose free directions (a common condition for those recently diagnosed with celiac)which is my usual prep method.  The only drawback of the lactose free version is the oil tends to separate from the batter as it bakes, but does not effect the overall quality of the finished product.  Leftovers should be refrigerated, and reheated in the microwave for best taste.</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113686</th>
      <td>1337817600</td>
      <td>Cheryl L. Malone</td>
      <td>Very Nice</td>
      <td>This seems to be a very good product.  What my husband really noticed about it is it doesn't strip the oil from the hair, like many other dandruff shampoos do.  It's a keeper.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>113687</th>
      <td>1137542400</td>
      <td>Lisa Gump "Home CEO"</td>
      <td>Wonderful afternoon treat!</td>
      <td>I did not buy this from this particular vendor, but I wanted to say this tea is really wonderful.  Just as described, it adds a nice vanilla and cream flavor to the earl grey.  I like it with a little sugar.  I bet it would be good with a little milk as well.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>113688</th>
      <td>1327190400</td>
      <td>T</td>
      <td>Great Food for Multi-Cat Household Also!</td>
      <td>I have seven cats and all of them do well with this product. No urinary troubles (one of my males had that issue with a different food). They also seem to vomit a LOT less less often from hairballs, the lack of which I attribute to the change in food. They've all been eating it for about a year now. I am back to order more which I do here on Amazon when the price is comparable or less than the local pet store, and I can also get free shipping.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>113689</th>
      <td>1325808000</td>
      <td>markw</td>
      <td>Gorgeous!</td>
      <td>I've never seen another schilleriana with such beautiful, bold foliage! This is a amazing hybrid and I can't wait to see the flowers!&lt;br /&gt;Also the plant was very healthy and had great roots.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>113690</th>
      <td>1301702400</td>
      <td>A. Damuth "build stuff!"</td>
      <td>Works well for our bread recipe.</td>
      <td>Our favorite yeast used to be some sold at the supermarket in little glass jars that are way too expensive.  We have been happy with the results of this Red Star yeast, and buying it 4 pounds at a time saves a lot of money.</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>113691 rows × 5 columns</p>
</div>



```python
#convert (int) timestamp to datetime
df1['Time'] = df1['Time'].apply(lambda x : datetime.datetime.fromtimestamp(x)) 
df1.head()
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
      <th>Time</th>
      <th>ProfileName</th>
      <th>Summary</th>
      <th>Text</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-09-13 08:00:00</td>
      <td>AMBER</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-11-04 08:00:00</td>
      <td>Connie Moreno</td>
      <td>Convenient but disappointing</td>
      <td>These taste just like any other packaged pork ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-02-08 08:00:00</td>
      <td>M. Abendroth</td>
      <td>One of the best</td>
      <td>For those who don't know, cacao nibs are small...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-03-18 08:00:00</td>
      <td>Maggie Leave "Maggie H"</td>
      <td>This is the best tea I've got in the U.S.</td>
      <td>Much better than other teas I've tried. Real t...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-06 08:00:00</td>
      <td>mom 94 "carin"</td>
      <td>very dense finished product</td>
      <td>I have purchased this item 3 times and have ha...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# see the summary of a dataset
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 113691 entries, 0 to 113690
    Data columns (total 5 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   Time         113691 non-null  datetime64[ns]
     1   ProfileName  113689 non-null  object        
     2   Summary      113684 non-null  object        
     3   Text         113691 non-null  object        
     4   Score        113691 non-null  int64         
    dtypes: datetime64[ns](1), int64(1), object(3)
    memory usage: 5.2+ MB
    


```python
# Because the wide range of score could make prediction more too challenging, catagrorized into two tiers: satisfied and not satisfied would help
# Create Sentiment Class
# Score 1-3: not satisfied
# Score 4-5: satisfied
df1['Satisfied'] = pd.cut(df1['Score'], bins =[0,3, float('inf')], labels =['not satisfied', 'satisfied'])
df1.iloc[::1000]

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
      <th>Time</th>
      <th>ProfileName</th>
      <th>Summary</th>
      <th>Text</th>
      <th>Score</th>
      <th>Satisfied</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-09-13 08:00:00</td>
      <td>AMBER</td>
      <td>Yay!</td>
      <td>Great deal! Great addition to a product that I...</td>
      <td>5</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>2012-06-28 08:00:00</td>
      <td>aka Shortie</td>
      <td>Fun game, not enough of each color to really play</td>
      <td>Got this as a gift for the office. It's nice t...</td>
      <td>3</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>2012-09-28 08:00:00</td>
      <td>Barbara De Roes "bderoes"</td>
      <td>more like canned beans than soup</td>
      <td>very tasty and filling.&lt;br /&gt;&lt;br /&gt;great value...</td>
      <td>5</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3000</th>
      <td>2011-03-16 08:00:00</td>
      <td>mren</td>
      <td>I like Maxwell House 100% Colombian Ground; to...</td>
      <td>This is a great price for a good coffee; but I...</td>
      <td>2</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>4000</th>
      <td>2010-02-06 08:00:00</td>
      <td>PennyK</td>
      <td>My dog loves these</td>
      <td>My dog has food allergies so it's hard to find...</td>
      <td>5</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109000</th>
      <td>2011-07-16 08:00:00</td>
      <td>gemboo</td>
      <td>My dogs love these!</td>
      <td>This is a favorite for one of my dogs!  When t...</td>
      <td>5</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>110000</th>
      <td>2012-09-06 08:00:00</td>
      <td>Sharon Rainey</td>
      <td>Completely addicting</td>
      <td>I can't keep these in my house, or I have to l...</td>
      <td>5</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>111000</th>
      <td>2006-06-11 08:00:00</td>
      <td>Mona Osman "zolwica"</td>
      <td>yum yum yum!</td>
      <td>They are more potato crockets in the shape of ...</td>
      <td>4</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>112000</th>
      <td>2011-11-25 08:00:00</td>
      <td>Peter Cook</td>
      <td>I ordered the wrong item....my fault.</td>
      <td>I wanted the Peppermint Breathsavers, not the ...</td>
      <td>4</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>113000</th>
      <td>2009-10-26 08:00:00</td>
      <td>Southern belle</td>
      <td>Caution, if you don't want a pocket full of po...</td>
      <td>I wish I had know that this product was change...</td>
      <td>2</td>
      <td>not satisfied</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 6 columns</p>
</div>




```python
#visualize the proportion of sample set with bar chart
ax = df1['Satisfied'].value_counts().plot(kind='bar',
                                    figsize=(8,8),
                                    title="Sentiment of Customer Extraced from Restaurant's reviews")
ax.set_xlabel("Sentiment of Customer")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_21_0.png)
    


# Data Preprocessing

Now, we will perform some pre-processing on the data before converting it into vectors and
passing it to the machine learning model.<br>

Objective: To reduce noise, which affect the accuracy rate of model prediction. Make it more simple for model to classify.<br>

Method:<br>
1) Using regular expresiion to get rid off any characters which are not alphabet and unnecssary<br>
2) convert the string to lowercase<br>
3) get rid off stopwords i.e 'the', 'an', 'to'; these are considres as noise which could make a model less precise<br>
4) lemmatization: chang different form of word i.e. working -> work <br>




```python
#Because this step taking a long time to generate, the cleaning text should be saved separately
#object of WordNetLemmatizer
#processing time: around 40 min
"""

lm = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item)) #match any characters which are not alphabet and replace with whitespace
        new_item = new_item.lower() # convert all to lower case
        new_item = new_item.split() # split each string by whitespace into a list
        # lemmarizing words & select only words which are not stopword in English 
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

corpus = text_transformation(df1['Text'])

"""


#Note: after cleaning text, there's some unwanted elements still
#so it's required to used regular expression to get rid of them (<br>)
"""
pattern0 = r'<br />'
clean = []
for i in df1.text_clean:
    a = re.sub(pattern0, ' ', i)
    clean.append(a)
    
pattern1 = r'<br>'
clean1= []
for i in clean:
    b = re.sub(pattern1, ' ', i)
    clean1.append(b)
    
    
pattern2 = r'\s(br)\s'
clean2= []
for i in clean1:
    c = re.sub(pattern2, ' ', i)
    clean2.append(c)    
    
"""



```


```python
#saveing the file
#df1.to_pickle("df1_clean.pkl")
```


```python
# showing some stopwords
print(stopwords.words('english'))
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    


```python
df1 = pd.read_pickle("df1_clean.pkl") # reading pkl file
```


```python
#show before and after cleaning
tmp =df1.iloc[::10000, [3, 6]]
with pd.option_context('display.max_colwidth', None):
  display(tmp)
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
      <th>Text</th>
      <th>text_clean</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Keeps the cups off the counter!! It is a very well made sturdy product. It is a little stiff to pull down but but I'd rather that than falling down by it's self.  We used the included screws to make sure it stays up. We are very pleased.</td>
      <td>keep cup counter well made sturdy product little stiff pull rather falling self used included screw make sure stay pleased</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>My son loves this food.  He is 16 months now and I still use them.  Not all the time, but often.  The reason is that his molars are coming in very quickly and he is in a lot of pain.  He won't eat much when he's in pain, but these are easy on his gums.  They are organic and a quick meal.  My son is strong and a very healthy weight.  I make sure he gets as much organic, wholesome food as possible.  Buying these subscribe and save is a great way to give him good food and still save money.  He loves the whole line.</td>
      <td>son love food month still use time often reason molar coming quickly lot pain eat much pain easy gum organic quick meal son strong healthy weight make sure get much organic wholesome food possible buying subscribe save great way give good food still save money love whole line</td>
    </tr>
    <tr>
      <th>20000</th>
      <td>Finally!! I love my Keurig, and I don't mind buying the k-cups. What does bother me is the limited availability of flavored decaf coffees. I am a coffee addict and could drink it all day if it didn't keep me up all night. I have tried every other do it yourself k cup product out there and they all stink! This product actually works. It brews a decent cup of coffee,  not watered down and not a single ground of coffee in the cup. This product is worth the money! Thinking about buying a second!</td>
      <td>finally love keurig mind buying k cup bother limited availability flavored decaf coffee coffee addict could drink day keep night tried every k cup product stink product actually work brew decent cup coffee watered single ground coffee cup product worth money thinking buying second</td>
    </tr>
    <tr>
      <th>30000</th>
      <td>I first discovered these several years ago on a trip to San Francisco, at the Rainbow Market. They are not only vegetarian and gluten-free, they have no preservatives. However, while authentic tasting, they are not quite spicy enough for me.&lt;br /&gt;After my young son decided to become a vegetarian, I started ordering this variety pack on a regular basis. Even though the price each is the same as it is in my local store, I can never find all six flavors at the same store at the same time, so it is worth it to order it this way. I serve these over basmati rice, and it's more than enough to feed two people.</td>
      <td>first discovered several year ago trip san francisco rainbow market vegetarian gluten free preservative however authentic tasting quite spicy enough young son decided become vegetarian started ordering variety pack regular basis even though price local store never find six flavor store time worth order way serve basmati rice enough feed two people</td>
    </tr>
    <tr>
      <th>40000</th>
      <td>The description on the 16oz Carousel-Sugarfree Gumball Refill Ordered from Candy Crate Inc. reads it contains a qty. of 900 gumballs. This is a lie- when you recieve this product there are only 114 servings in a bag and the serving size is considered 2 gumballs- this is a total of only 228 gumballs. To get the quantity they are claiming you would actually have to purchase about 4 bags. The gumballs themselves are fine, but beware of the fake description. Hard to write a good review when I feel a little ripped off. When purchasing online all we have to go by are the descriptions - if they are not accurate what can we base our purchase on?</td>
      <td>description oz carousel sugarfree gumball refill ordered candy crate inc read contains qty gumballs lie recieve product serving bag serving size considered gumballs total gumballs get quantity claiming would actually purchase bag gumballs fine beware fake description hard write good review feel little ripped purchasing online go description accurate base purchase</td>
    </tr>
    <tr>
      <th>50000</th>
      <td>Although I have never gotten it through Amazon.com, baklava made by Shatila Food Products in Michigan (www.shatila.com) is the best there is!  Forget the stuff you find at the local bakery or Harry &amp; David--this stuff is in a different league altogether.</td>
      <td>although never gotten amazon com baklava made shatila food product michigan www shatila com best forget stuff find local bakery harry david stuff different league altogether</td>
    </tr>
    <tr>
      <th>60000</th>
      <td>I have updated this mixed review due to the currently outrageous high shipping does NOT justify this product's value. Although This is the best gluten free sour dough bread I have sampled thus far it is much too expensive to pay upwards to $30 for 3 loaves.  I have shipped baked goods to military family priority mail for a flat rate of under ten bucks in the past.&lt;br /&gt;Company states they will reduce shipping charges in the future but that doesn't help me or others currently eager for this product.&lt;br /&gt;&lt;br /&gt;As for the quality/consistency of this bread, It is fluffy compared to ther bakeries and has not molded after 10 days.  It toasts well and easily makes grilled sandwiches delicious.  The nutritional content is as follows :  1 slice is 140 calories, no saturated or trans fats, total fat 2 grams.  Cholesterol 20 mg, sodium 190 mg, total carb ( for one huge slice ) is 29 grams, fiber 1 gram , sugars 1 gram , protein 2 grams.  This has definetly a great sour dough flavor and is not gritty like other gluten free baked goods from other bakeries..I only wish the shipping costs were less with this and all other gluten free baked goods.</td>
      <td>updated mixed review due currently outrageous high shipping justify product value although best gluten free sour dough bread sampled thus far much expensive pay upwards loaf shipped baked good military family priority mail flat rate ten buck past company state reduce shipping charge future help others currently eager product quality consistency bread fluffy compared ther bakery molded day toast well easily make grilled sandwich delicious nutritional content follows slice calorie saturated trans fat total fat gram cholesterol mg sodium mg total carb one huge slice gram fiber gram sugar gram protein gram definetly great sour dough flavor gritty like gluten free baked good bakery wish shipping cost le gluten free baked good</td>
    </tr>
    <tr>
      <th>70000</th>
      <td>I don't have a gluten intolerance - just trying to cut back on the intake of wheat/gluten...and sugar, but that's another story;) so my body feels less bloated, unhealthy and lethargic due to wheat.  With that said, I do know the differences in taste between wheat pastas, rice, quinoa, corn...etc.  Personally, I have come to love the taste of non-wheat pastas over wheat, with the exception of corn flour, which isn't similar enough to wheat to fool my taste buds.  Annie's does a fantastic job with this rice flour product, giving it a consistency and flavor akin to Kraft mac and cheese.  What makes this better is the cheese, which tastes far yummier than any boxed mac &amp; cheese I have ever had.  Also, it's real cheese, with as few bizarre ingredients as possible.&lt;br /&gt;&lt;br /&gt;For those who like to see ingredients, here is a comparison (and please note I am not a nutritionist - just writing a friendly review:):&lt;br /&gt;&lt;br /&gt;Kraft Mac &amp; Cheese:&lt;br /&gt;Cheese sauce mix ingredients: whey (milk protein), milk protein concentrate, milk, milkfat and cheese culture, salt, sodium tripolyphosphate, sodium phosphate and calcium phosphate, Yellow 5 and Yellow 6, citric acid, lactic acid and enzymes.&lt;br /&gt;&lt;br /&gt;Annie's Rice Pasta &amp; Cheddar: cheddar cheese (cultured pasturized milk, salt, non-animal enzymes), whey, buttermilk, salt, cream, natural flavor, natural sodium phosphate, annatto extract for natural color.&lt;br /&gt;&lt;br /&gt;*wiki annatto extract: Annatto coloring is produced from the reddish pericarp or pulp which surrounds the seed of the achiote (Bixa orellana L.). It is used in many natural cheeses (e.g., Cheddar, Red Leicester, Gouda (cheese) and Brie), margarine, butter, rice, smoked fish, and custard powder.&lt;br /&gt;&lt;br /&gt;Annie's also has less sodium and sugars, which I am grateful for.&lt;br /&gt;&lt;br /&gt;Also, Annie's does make another rice pasta mac and cheese - it's a deluxe box.  This is what I would compare to Velveeta - for you lovers out there.  It's the ooey gooey cheese that is thicker.  Personally, I detest Velveeta, so the deluxe isn't as awesome as the simple Rice Pasta &amp; Cheddar.  But the deluxe IS better than macaroni with Velveeta because the consistency of the cheese isn't ridiculous overbearing and throat-clogging as Velveeta.  I swear, I always felt like I would suffocate eating that stuff!&lt;br /&gt;&lt;br /&gt;I definitely recommend this product to those with allergies, and intolerance, or those like myself who are looking for ways to significantly reduce heat intake.  My entire family has now switched from Kraft over to this product (and they didn't do it for health reasons - they simply prefer the taste!)&lt;br /&gt;&lt;br /&gt;It's a bit more pricey, I'll give you that.  But for a hint - do check Target occasionally.  They sell Annie's pastas and some amazing organic bunny fruit snacks - all of which go on sale quite often (I just purchased Rice Pasta &amp; Cheddar for $1 a box!)  If only the prices were always so kind;)</td>
      <td>gluten intolerance trying cut back intake wheat gluten sugar another story body feel le bloated unhealthy lethargic due wheat said know difference taste wheat pasta rice quinoa corn etc personally come love taste non wheat pasta wheat exception corn flour similar enough wheat fool taste bud annie fantastic job rice flour product giving consistency flavor akin kraft mac cheese make better cheese taste far yummier boxed mac cheese ever also real cheese bizarre ingredient possible like see ingredient comparison please note nutritionist writing friendly review kraft mac cheese cheese sauce mix ingredient whey milk protein milk protein concentrate milk milkfat cheese culture salt sodium tripolyphosphate sodium phosphate calcium phosphate yellow yellow citric acid lactic acid enzyme annie rice pasta cheddar cheddar cheese cultured pasturized milk salt non animal enzyme whey buttermilk salt cream natural flavor natural sodium phosphate annatto extract natural color wiki annatto extract annatto coloring produced reddish pericarp pulp surround seed achiote bixa orellana l used many natural cheese e g cheddar red leicester gouda cheese brie margarine butter rice smoked fish custard powder annie also le sodium sugar grateful also annie make another rice pasta mac cheese deluxe box would compare velveeta lover ooey gooey cheese thicker personally detest velveeta deluxe awesome simple rice pasta cheddar deluxe better macaroni velveeta consistency cheese ridiculous overbearing throat clogging velveeta swear always felt like would suffocate eating stuff definitely recommend product allergy intolerance like looking way significantly reduce heat intake entire family switched kraft product health reason simply prefer taste bit pricey give hint check target occasionally sell annie pasta amazing organic bunny fruit snack go sale quite often purchased rice pasta cheddar box price always kind</td>
    </tr>
    <tr>
      <th>80000</th>
      <td>This drink tastes good.  I enjoyed it. I also had my daughter and my grandchildren try it--they drank even more of it and found it pleasant to the taste.  Once mixed and cold in the fridge, it didn't last long.  A good alternative to pop I think.&lt;br /&gt;&lt;br /&gt;Recommended.</td>
      <td>drink taste good enjoyed also daughter grandchild try drank even found pleasant taste mixed cold fridge last long good alternative pop think recommended</td>
    </tr>
    <tr>
      <th>90000</th>
      <td>I quite enjoyed these cookies.  They are reminicent of a shortbread cookie with a hint of orange and some chewy Crasins thrown in for good measure.  Fairly reasonable stats for a cookie (140 calories, 5 grams of total fat and 7 sugars)---until you see that is only for THREE cookies.  No way you're going to hold yourself to 3 lousy cookies in one sitting so you'd better plan on doubling that.  But it is still a better choice that a lot of offerings in the cookie isle.  And that is just where I'd go to purchase these.  They didn't hold up well in shipping and I wound up with a lot of crumbs.  Which I ate anyway.  Because they were too yummy to let that stop me.  Enjoy!</td>
      <td>quite enjoyed cooky reminicent shortbread cookie hint orange chewy crasins thrown good measure fairly reasonable stats cookie calorie gram total fat sugar see three cooky way going hold lousy cooky one sitting better plan doubling still better choice lot offering cookie isle go purchase hold well shipping wound lot crumb ate anyway yummy let stop enjoy</td>
    </tr>
    <tr>
      <th>100000</th>
      <td>A very nutritrious and delicious soup from Amy's not offered in the organic sections of grocery stores in my area of the U.S. But about a quarter of the cans in the case were dented, and therefore, not acceptable for long term storage.&lt;br /&gt; I wouldn't buy a dented can from a store, and therefore am dismayed that Amazon would ship damaged goods.&lt;br /&gt; If Amazon is getting a good price on this product because the cans are dented already, the product should be advertized as such. I don't like being sent a case of canned goods with the cans in the middle of the case crushed. What's up with that?&lt;br /&gt; I have been satisfied with the condition of other canned goods bought vie Amazon - but buyer beware.&lt;br /&gt; It is not worth my time to complain and return.&lt;br /&gt; But I won't buy Amy's soups through Amazon again, and in the future, will REALLY question whether buying ANY canned goods through Amazon is worth it - even if the price is right -  considering that the goods may or may not arrive damaged.&lt;br /&gt; Hey Amazon, Honesty is the best policy. It's not a "deal" if you send me damaged goods.</td>
      <td>nutritrious delicious soup amy offered organic section grocery store area u quarter can case dented therefore acceptable long term storage buy dented store therefore dismayed amazon would ship damaged good amazon getting good price product can dented already product advertized like sent case canned good can middle case crushed satisfied condition canned good bought vie amazon buyer beware worth time complain return buy amy soup amazon future really question whether buying canned good amazon worth even price right considering good may may arrive damaged hey amazon honesty best policy deal send damaged good</td>
    </tr>
    <tr>
      <th>110000</th>
      <td>There are several things a coffee lover looks for in their brew.. the aroma, the color and the taste are what I look for. When I opened the individual pack, I was hit with a wonderful coffee scent.  The pod looks typical, and are made for a pod machine.  When I made my first cup, since the aroma was strong, I filled the machine with a good sized mug's worth of water, and made a cup.  When it was done, the color was fairly light, so I only added a small amount of milk.  Still, the flavor was a bit too bland for me, and I like a mild coffee.  For the second cup, I used a smaller mug, and in return got a darker, more flavorful cup, so that is my recommendation with this brand.  Some other things I love about this brand:  it's organic, sustainably grown and Fair Trade certified. That's a lot of benefits for only about 75 cents a cup.  The even have their own foundation to support youth soccer programs.  This is a good deal.  The only thing that I would change is the individual wrap, which seems like an excess of packaging for a company dedicated to the environment.</td>
      <td>several thing coffee lover look brew aroma color taste look opened individual pack hit wonderful coffee scent pod look typical made pod machine made first cup since aroma strong filled machine good sized mug worth water made cup done color fairly light added small amount milk still flavor bit bland like mild coffee second cup used smaller mug return got darker flavorful cup recommendation brand thing love brand organic sustainably grown fair trade certified lot benefit cent cup even foundation support youth soccer program good deal thing would change individual wrap seems like excess packaging company dedicated environment</td>
    </tr>
  </tbody>
</table>
</div>


# WordClound
- using wordclound to find the most frequency of word being used in review<br>
- it is required to convert pandas data serie (text_clean column) into a long string in a variable<br>

note that the result is just the long string in a variabal, which we need to pass that to a wordclund object






```python
"""
#Processing Time: 10 min
#preparing data for wordcloud visualization
word = df1['text_clean']
comment_words = ""  # create empty string variable

i=0
j=0

#loop to each row in corpus and append them to comment_words variable
while j <= len(word)-1: #setting number of counter equal to number of observation -1, otherwise, out of inde
    i = word[j]
    comment_words +="".join(i) # for each word append into comment_words variable
    j = j+1 # increae the counter
"""


```


```python
#len(comment_words)
```


```python
#type(comment_words)
```


```python
#comment_words[0:1000]
```


```python
# passing all parameter and 'comment_words' variable, which we generate from previous step
"""
wordcloud = WordCloud(width = 1500, height = 1500,background_color ='white',min_font_size = 10).generate(comment_words)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('High Frequency of Words Found in Customer Reviews')
plt.savefig('wordclound.png') # set the file to png.
"""

```




    "\nwordcloud = WordCloud(width = 1500, height = 1500,background_color ='white',min_font_size = 10).generate(comment_words)\nplt.figure(figsize=(15, 10))\nplt.imshow(wordcloud)\nplt.title('High Frequency of Words Found in Customer Reviews')\nplt.savefig('wordclound.png') # set the file to png.\n"



![wordclound.png](attachment:wordclound.png)

do the visualization with heatmap 
Assumption: different score review should have different position in vector space so we will 
utilize heat map to answer the question that the reviews with different range of score are really different in vector space

### A short note of what is Word Embedding
Word Embedding
Word Embeddings are the texts converted into numbers and there may be different numerical representations of the same text
In short, we can say that to build any model in machine learning or deep learning, the final level data has to be in numerical form because models don’t understand text or image data directly as humans do.
Therefore, Vectorization or word embedding is the process of converting text data to numerical vectors. Later those vectors are used to build various machine learning models. In this manner, we say this as extracting features with the help of text with an aim to build multiple natural languages, processing models, etc. We have different ways to convert the text data to numerical vectors which we will discuss in this article later.
Broadly, we can classified word embeddings into the following two categories:
Frequency-based or Statistical based Word Embedding
Prediction based Word Embedding





```python
# catagorize reviews into two groups: score 4 and 5, score <= 3
```


```python
#filter only text_clen which score = 5 
filter0 = df1['Score'] == 5
score_5 = df1[filter0]

#filter only text_clen which score <4
filter1 = df1['Score'] < 4
score_1to3 = df1[filter1]
```


```python
score_5 = score_5[['Time','ProfileName','text_clean','Score']].iloc[0:500] #must be in the same shape
```


```python
score_1to3 = score_1to3[['Time','ProfileName','text_clean','Score']].iloc[0:500] #must be in the same shape
```


```python
tf_score5 = score_5
```


```python
score_5
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
      <th>Time</th>
      <th>ProfileName</th>
      <th>text_clean</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-02-11 08:00:00</td>
      <td>cac Idaho</td>
      <td>keep cup counter well made sturdy product litt...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-10-16 08:00:00</td>
      <td>Auskan "Auskan"</td>
      <td>love pantry cook batch rice add sauce dinner s...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-08-24 08:00:00</td>
      <td>chicago</td>
      <td>used another brand tonkotsu flavor noodle impo...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-07-13 08:00:00</td>
      <td>you suckkk</td>
      <td>herr favorite chip brand fan salsa love chip</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012-03-15 08:00:00</td>
      <td>Donna</td>
      <td>absolutely good french vanilla cappuccino boug...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>787</th>
      <td>2010-12-09 08:00:00</td>
      <td>Erika</td>
      <td>new favorite snack food whenever craving sweet...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>789</th>
      <td>2007-03-09 08:00:00</td>
      <td>J. Lamar</td>
      <td>prepared kit basic add shrimp anything red pep...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>792</th>
      <td>2011-11-21 08:00:00</td>
      <td>JVR Mom</td>
      <td>month old daughter love formula mixing issue t...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>793</th>
      <td>2011-01-25 08:00:00</td>
      <td>Stacy "sllemke"</td>
      <td>best ever candy person sweet general however f...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>795</th>
      <td>2009-03-03 08:00:00</td>
      <td>O. Vinogradova "jaded mouse"</td>
      <td>love treat training small tasty least dog seem...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>
</div>




```python
score_1to3
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
      <th>Time</th>
      <th>ProfileName</th>
      <th>text_clean</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>2011-03-12 08:00:00</td>
      <td>CANDICE</td>
      <td>pop nice never get taste like movie theater po...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2011-08-25 08:00:00</td>
      <td>Light by the Moon</td>
      <td>ordered birthday got birthday money family ord...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2009-05-25 08:00:00</td>
      <td>MamavanMNE</td>
      <td>candy good taste seem made natural ingredient ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2011-05-11 08:00:00</td>
      <td>Dr. M. A. Dixon "hyper-observant"</td>
      <td>tea taste like blend ingredient listed taste l...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2012-06-15 08:00:00</td>
      <td>annie "grannieannie"</td>
      <td>put enough creamer coffee tolerable good coffe...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2221</th>
      <td>2009-09-29 08:00:00</td>
      <td>Robert Y. Lamaute "blamaute"</td>
      <td>light bright florescent bulb wattage look nice...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2246</th>
      <td>2012-04-03 08:00:00</td>
      <td>Lindsay Pasch "VaBookworm87"</td>
      <td>come conclusion big fan thing definitely say m...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2250</th>
      <td>2011-02-08 08:00:00</td>
      <td>Robert C. Reade "Random buyer"</td>
      <td>ordered coffee another brand seems three week ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2259</th>
      <td>2006-11-10 08:00:00</td>
      <td>Kate</td>
      <td>bought amazon becuase disappeared real store c...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2260</th>
      <td>2010-07-18 08:00:00</td>
      <td>Steven Meuse</td>
      <td>lucky stock canister last summer three left wr...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>
</div>



# Transform Text to Vector


```python
# transform those text into vectors, which is actuall appeared in sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
tf_score5  = count_vect.fit_transform(score_5['text_clean'])
tf_score5
tf_score1to3  = count_vect.fit_transform(score_1to3['text_clean'])
tf_score1to3

```




    <500x4287 sparse matrix of type '<class 'numpy.int64'>'
    	with 18782 stored elements in Compressed Sparse Row format>




```python
# check its shape
tf_score1to3.shape
```




    (500, 4287)



# Cosine Similarity

After some kind of transforming text to vector, we need to reshape sparse matrix
so we can use a coins_similarity function to generate its cosine similarity.
Cosine Similarity is one of the method to measure the distance of different data points in vector space
and , in our case, we will implement that and visualize cosine similarity of those reviews with heat map.

## Reshape Sparse matrix


```python
tf_score5=tf_score5[0:500, 0:3511].toarray() #reshape sparse matrix
```


```python
tf_score1to3=tf_score1to3[0:500, 0:3511].toarray()#reshape sparse matrix
```


```python
tf_score1to3
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)



## Get the Cosine Score


```python
#get the cosine score
from sklearn.metrics.pairwise import cosine_similarity 
cosinescore = cosine_similarity(tf_score5 ,tf_score1to3)
```


```python
cosinescore
```




    array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.04783649, 0.        ,
            0.03181424],
           [0.        , 0.01756821, 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.02179068, 0.03846154, ..., 0.        , 0.        ,
            0.        ]])



# Heatmap


```python
# the size on data which will be displayed in heat map
plot_z = cosinescore[0:40, 0:40]
```


```python
# generate heat map of review 1-3 VS 4-5 scores
import seaborn as sns

df_todraw = pd.DataFrame(plot_z)
plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_todraw,
                 cmap="YlGnBu",
                 vmin=0, vmax=1, annot=True, fmt='.1f')
plt.show()

#Note: the heat map showing us that there's less similarity between these two group of review, which it is supposed to be like that
#because they have significanet different range of score.
```


    
![png](output_59_0.png)
    



```python
#compare score 5 to another 5 score  review
```


```python
cosinescore5 = cosine_similarity(tf_score5 ,tf_score5)
```


```python
cosinescore5
```




    array([[1.        , 0.        , 0.10606602, ..., 0.05      , 0.        ,
            0.0438529 ],
           [0.        , 1.        , 0.10882144, ..., 0.15389675, 0.086711  ,
            0.08998425],
           [0.10606602, 0.10882144, 1.        , ..., 0.10606602, 0.02988072,
            0.09302605],
           ...,
           [0.05      , 0.15389675, 0.10606602, ..., 1.        , 0.08451543,
            0.0438529 ],
           [0.        , 0.086711  , 0.02988072, ..., 0.08451543, 1.        ,
            0.03706247],
           [0.0438529 , 0.08998425, 0.09302605, ..., 0.0438529 , 0.03706247,
            1.        ]])




```python
plot_zz = cosinescore5[0:40, 41:81]
```


```python
plot_x = list(range(41,81))
```


```python
import seaborn as sns

df_todraw2 = pd.DataFrame(plot_zz, columns = plot_x)
plt.subplots(figsize=(20, 15))
ax = sns.heatmap(df_todraw2,
                 cmap="YlGnBu",
                 vmin=0, vmax=1, annot=True, fmt='.1f')
plt.show()

#Note: While comparing between those review with only 5 score, they show the likelihood of being similar more.
```


    
![png](output_65_0.png)
    



```python
#pulling out some review which has high correlation and see how they being similar
```


```python
score_5.iloc[3,2]
```




    'herr favorite chip brand fan salsa love chip'




```python
score_5.iloc[60,2]
```




    'cannot tolerate extremely hot spicy chip like zing crunch chip made enjoyment purchased around holiday truly enjoyed many guest yes purchasing chip really good organic affordable'




```python
score_5.iloc[33,2]
```




    'love arizona green tea ginseng drink time simple easy carry packet purse'




```python
score_5.iloc[78,2]
```




    'love love green tea hard find area place internet charge big price usually get many box merchant definitely order seller thanks depend green tea fix everyday'




```python
score_5.iloc[36,2]
```




    'cereal tasty healthy spice bit good add banana walnut coconut shaving mmm good'




```python
score_5.iloc[37,2] # score 0.0, this review is about cereal
```




    'love chip longer crave regular potato chip tasty crunchy alot salt tho'



Result: those pairs ,which receive cosine similarity at 0.4 and 0.3, are all good revew about chip


```python
#displaying some data
tmp =df1.iloc[::10000, [3, 6]]
with pd.option_context('display.max_colwidth', None):
  display(tmp)
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
      <th>Text</th>
      <th>text_clean</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Keeps the cups off the counter!! It is a very well made sturdy product. It is a little stiff to pull down but but I'd rather that than falling down by it's self.  We used the included screws to make sure it stays up. We are very pleased.</td>
      <td>keep cup counter well made sturdy product little stiff pull rather falling self used included screw make sure stay pleased</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>My son loves this food.  He is 16 months now and I still use them.  Not all the time, but often.  The reason is that his molars are coming in very quickly and he is in a lot of pain.  He won't eat much when he's in pain, but these are easy on his gums.  They are organic and a quick meal.  My son is strong and a very healthy weight.  I make sure he gets as much organic, wholesome food as possible.  Buying these subscribe and save is a great way to give him good food and still save money.  He loves the whole line.</td>
      <td>son love food month still use time often reason molar coming quickly lot pain eat much pain easy gum organic quick meal son strong healthy weight make sure get much organic wholesome food possible buying subscribe save great way give good food still save money love whole line</td>
    </tr>
    <tr>
      <th>20000</th>
      <td>Finally!! I love my Keurig, and I don't mind buying the k-cups. What does bother me is the limited availability of flavored decaf coffees. I am a coffee addict and could drink it all day if it didn't keep me up all night. I have tried every other do it yourself k cup product out there and they all stink! This product actually works. It brews a decent cup of coffee,  not watered down and not a single ground of coffee in the cup. This product is worth the money! Thinking about buying a second!</td>
      <td>finally love keurig mind buying k cup bother limited availability flavored decaf coffee coffee addict could drink day keep night tried every k cup product stink product actually work brew decent cup coffee watered single ground coffee cup product worth money thinking buying second</td>
    </tr>
    <tr>
      <th>30000</th>
      <td>I first discovered these several years ago on a trip to San Francisco, at the Rainbow Market. They are not only vegetarian and gluten-free, they have no preservatives. However, while authentic tasting, they are not quite spicy enough for me.&lt;br /&gt;After my young son decided to become a vegetarian, I started ordering this variety pack on a regular basis. Even though the price each is the same as it is in my local store, I can never find all six flavors at the same store at the same time, so it is worth it to order it this way. I serve these over basmati rice, and it's more than enough to feed two people.</td>
      <td>first discovered several year ago trip san francisco rainbow market vegetarian gluten free preservative however authentic tasting quite spicy enough young son decided become vegetarian started ordering variety pack regular basis even though price local store never find six flavor store time worth order way serve basmati rice enough feed two people</td>
    </tr>
    <tr>
      <th>40000</th>
      <td>The description on the 16oz Carousel-Sugarfree Gumball Refill Ordered from Candy Crate Inc. reads it contains a qty. of 900 gumballs. This is a lie- when you recieve this product there are only 114 servings in a bag and the serving size is considered 2 gumballs- this is a total of only 228 gumballs. To get the quantity they are claiming you would actually have to purchase about 4 bags. The gumballs themselves are fine, but beware of the fake description. Hard to write a good review when I feel a little ripped off. When purchasing online all we have to go by are the descriptions - if they are not accurate what can we base our purchase on?</td>
      <td>description oz carousel sugarfree gumball refill ordered candy crate inc read contains qty gumballs lie recieve product serving bag serving size considered gumballs total gumballs get quantity claiming would actually purchase bag gumballs fine beware fake description hard write good review feel little ripped purchasing online go description accurate base purchase</td>
    </tr>
    <tr>
      <th>50000</th>
      <td>Although I have never gotten it through Amazon.com, baklava made by Shatila Food Products in Michigan (www.shatila.com) is the best there is!  Forget the stuff you find at the local bakery or Harry &amp; David--this stuff is in a different league altogether.</td>
      <td>although never gotten amazon com baklava made shatila food product michigan www shatila com best forget stuff find local bakery harry david stuff different league altogether</td>
    </tr>
    <tr>
      <th>60000</th>
      <td>I have updated this mixed review due to the currently outrageous high shipping does NOT justify this product's value. Although This is the best gluten free sour dough bread I have sampled thus far it is much too expensive to pay upwards to $30 for 3 loaves.  I have shipped baked goods to military family priority mail for a flat rate of under ten bucks in the past.&lt;br /&gt;Company states they will reduce shipping charges in the future but that doesn't help me or others currently eager for this product.&lt;br /&gt;&lt;br /&gt;As for the quality/consistency of this bread, It is fluffy compared to ther bakeries and has not molded after 10 days.  It toasts well and easily makes grilled sandwiches delicious.  The nutritional content is as follows :  1 slice is 140 calories, no saturated or trans fats, total fat 2 grams.  Cholesterol 20 mg, sodium 190 mg, total carb ( for one huge slice ) is 29 grams, fiber 1 gram , sugars 1 gram , protein 2 grams.  This has definetly a great sour dough flavor and is not gritty like other gluten free baked goods from other bakeries..I only wish the shipping costs were less with this and all other gluten free baked goods.</td>
      <td>updated mixed review due currently outrageous high shipping justify product value although best gluten free sour dough bread sampled thus far much expensive pay upwards loaf shipped baked good military family priority mail flat rate ten buck past company state reduce shipping charge future help others currently eager product quality consistency bread fluffy compared ther bakery molded day toast well easily make grilled sandwich delicious nutritional content follows slice calorie saturated trans fat total fat gram cholesterol mg sodium mg total carb one huge slice gram fiber gram sugar gram protein gram definetly great sour dough flavor gritty like gluten free baked good bakery wish shipping cost le gluten free baked good</td>
    </tr>
    <tr>
      <th>70000</th>
      <td>I don't have a gluten intolerance - just trying to cut back on the intake of wheat/gluten...and sugar, but that's another story;) so my body feels less bloated, unhealthy and lethargic due to wheat.  With that said, I do know the differences in taste between wheat pastas, rice, quinoa, corn...etc.  Personally, I have come to love the taste of non-wheat pastas over wheat, with the exception of corn flour, which isn't similar enough to wheat to fool my taste buds.  Annie's does a fantastic job with this rice flour product, giving it a consistency and flavor akin to Kraft mac and cheese.  What makes this better is the cheese, which tastes far yummier than any boxed mac &amp; cheese I have ever had.  Also, it's real cheese, with as few bizarre ingredients as possible.&lt;br /&gt;&lt;br /&gt;For those who like to see ingredients, here is a comparison (and please note I am not a nutritionist - just writing a friendly review:):&lt;br /&gt;&lt;br /&gt;Kraft Mac &amp; Cheese:&lt;br /&gt;Cheese sauce mix ingredients: whey (milk protein), milk protein concentrate, milk, milkfat and cheese culture, salt, sodium tripolyphosphate, sodium phosphate and calcium phosphate, Yellow 5 and Yellow 6, citric acid, lactic acid and enzymes.&lt;br /&gt;&lt;br /&gt;Annie's Rice Pasta &amp; Cheddar: cheddar cheese (cultured pasturized milk, salt, non-animal enzymes), whey, buttermilk, salt, cream, natural flavor, natural sodium phosphate, annatto extract for natural color.&lt;br /&gt;&lt;br /&gt;*wiki annatto extract: Annatto coloring is produced from the reddish pericarp or pulp which surrounds the seed of the achiote (Bixa orellana L.). It is used in many natural cheeses (e.g., Cheddar, Red Leicester, Gouda (cheese) and Brie), margarine, butter, rice, smoked fish, and custard powder.&lt;br /&gt;&lt;br /&gt;Annie's also has less sodium and sugars, which I am grateful for.&lt;br /&gt;&lt;br /&gt;Also, Annie's does make another rice pasta mac and cheese - it's a deluxe box.  This is what I would compare to Velveeta - for you lovers out there.  It's the ooey gooey cheese that is thicker.  Personally, I detest Velveeta, so the deluxe isn't as awesome as the simple Rice Pasta &amp; Cheddar.  But the deluxe IS better than macaroni with Velveeta because the consistency of the cheese isn't ridiculous overbearing and throat-clogging as Velveeta.  I swear, I always felt like I would suffocate eating that stuff!&lt;br /&gt;&lt;br /&gt;I definitely recommend this product to those with allergies, and intolerance, or those like myself who are looking for ways to significantly reduce heat intake.  My entire family has now switched from Kraft over to this product (and they didn't do it for health reasons - they simply prefer the taste!)&lt;br /&gt;&lt;br /&gt;It's a bit more pricey, I'll give you that.  But for a hint - do check Target occasionally.  They sell Annie's pastas and some amazing organic bunny fruit snacks - all of which go on sale quite often (I just purchased Rice Pasta &amp; Cheddar for $1 a box!)  If only the prices were always so kind;)</td>
      <td>gluten intolerance trying cut back intake wheat gluten sugar another story body feel le bloated unhealthy lethargic due wheat said know difference taste wheat pasta rice quinoa corn etc personally come love taste non wheat pasta wheat exception corn flour similar enough wheat fool taste bud annie fantastic job rice flour product giving consistency flavor akin kraft mac cheese make better cheese taste far yummier boxed mac cheese ever also real cheese bizarre ingredient possible like see ingredient comparison please note nutritionist writing friendly review kraft mac cheese cheese sauce mix ingredient whey milk protein milk protein concentrate milk milkfat cheese culture salt sodium tripolyphosphate sodium phosphate calcium phosphate yellow yellow citric acid lactic acid enzyme annie rice pasta cheddar cheddar cheese cultured pasturized milk salt non animal enzyme whey buttermilk salt cream natural flavor natural sodium phosphate annatto extract natural color wiki annatto extract annatto coloring produced reddish pericarp pulp surround seed achiote bixa orellana l used many natural cheese e g cheddar red leicester gouda cheese brie margarine butter rice smoked fish custard powder annie also le sodium sugar grateful also annie make another rice pasta mac cheese deluxe box would compare velveeta lover ooey gooey cheese thicker personally detest velveeta deluxe awesome simple rice pasta cheddar deluxe better macaroni velveeta consistency cheese ridiculous overbearing throat clogging velveeta swear always felt like would suffocate eating stuff definitely recommend product allergy intolerance like looking way significantly reduce heat intake entire family switched kraft product health reason simply prefer taste bit pricey give hint check target occasionally sell annie pasta amazing organic bunny fruit snack go sale quite often purchased rice pasta cheddar box price always kind</td>
    </tr>
    <tr>
      <th>80000</th>
      <td>This drink tastes good.  I enjoyed it. I also had my daughter and my grandchildren try it--they drank even more of it and found it pleasant to the taste.  Once mixed and cold in the fridge, it didn't last long.  A good alternative to pop I think.&lt;br /&gt;&lt;br /&gt;Recommended.</td>
      <td>drink taste good enjoyed also daughter grandchild try drank even found pleasant taste mixed cold fridge last long good alternative pop think recommended</td>
    </tr>
    <tr>
      <th>90000</th>
      <td>I quite enjoyed these cookies.  They are reminicent of a shortbread cookie with a hint of orange and some chewy Crasins thrown in for good measure.  Fairly reasonable stats for a cookie (140 calories, 5 grams of total fat and 7 sugars)---until you see that is only for THREE cookies.  No way you're going to hold yourself to 3 lousy cookies in one sitting so you'd better plan on doubling that.  But it is still a better choice that a lot of offerings in the cookie isle.  And that is just where I'd go to purchase these.  They didn't hold up well in shipping and I wound up with a lot of crumbs.  Which I ate anyway.  Because they were too yummy to let that stop me.  Enjoy!</td>
      <td>quite enjoyed cooky reminicent shortbread cookie hint orange chewy crasins thrown good measure fairly reasonable stats cookie calorie gram total fat sugar see three cooky way going hold lousy cooky one sitting better plan doubling still better choice lot offering cookie isle go purchase hold well shipping wound lot crumb ate anyway yummy let stop enjoy</td>
    </tr>
    <tr>
      <th>100000</th>
      <td>A very nutritrious and delicious soup from Amy's not offered in the organic sections of grocery stores in my area of the U.S. But about a quarter of the cans in the case were dented, and therefore, not acceptable for long term storage.&lt;br /&gt; I wouldn't buy a dented can from a store, and therefore am dismayed that Amazon would ship damaged goods.&lt;br /&gt; If Amazon is getting a good price on this product because the cans are dented already, the product should be advertized as such. I don't like being sent a case of canned goods with the cans in the middle of the case crushed. What's up with that?&lt;br /&gt; I have been satisfied with the condition of other canned goods bought vie Amazon - but buyer beware.&lt;br /&gt; It is not worth my time to complain and return.&lt;br /&gt; But I won't buy Amy's soups through Amazon again, and in the future, will REALLY question whether buying ANY canned goods through Amazon is worth it - even if the price is right -  considering that the goods may or may not arrive damaged.&lt;br /&gt; Hey Amazon, Honesty is the best policy. It's not a "deal" if you send me damaged goods.</td>
      <td>nutritrious delicious soup amy offered organic section grocery store area u quarter can case dented therefore acceptable long term storage buy dented store therefore dismayed amazon would ship damaged good amazon getting good price product can dented already product advertized like sent case canned good can middle case crushed satisfied condition canned good bought vie amazon buyer beware worth time complain return buy amy soup amazon future really question whether buying canned good amazon worth even price right considering good may may arrive damaged hey amazon honesty best policy deal send damaged good</td>
    </tr>
    <tr>
      <th>110000</th>
      <td>There are several things a coffee lover looks for in their brew.. the aroma, the color and the taste are what I look for. When I opened the individual pack, I was hit with a wonderful coffee scent.  The pod looks typical, and are made for a pod machine.  When I made my first cup, since the aroma was strong, I filled the machine with a good sized mug's worth of water, and made a cup.  When it was done, the color was fairly light, so I only added a small amount of milk.  Still, the flavor was a bit too bland for me, and I like a mild coffee.  For the second cup, I used a smaller mug, and in return got a darker, more flavorful cup, so that is my recommendation with this brand.  Some other things I love about this brand:  it's organic, sustainably grown and Fair Trade certified. That's a lot of benefits for only about 75 cents a cup.  The even have their own foundation to support youth soccer programs.  This is a good deal.  The only thing that I would change is the individual wrap, which seems like an excess of packaging for a company dedicated to the environment.</td>
      <td>several thing coffee lover look brew aroma color taste look opened individual pack hit wonderful coffee scent pod look typical made pod machine made first cup since aroma strong filled machine good sized mug worth water made cup done color fairly light added small amount milk still flavor bit bland like mild coffee second cup used smaller mug return got darker flavorful cup recommendation brand thing love brand organic sustainably grown fair trade certified lot benefit cent cup even foundation support youth soccer program good deal thing would change individual wrap seems like excess packaging company dedicated environment</td>
    </tr>
  </tbody>
</table>
</div>



```python
#showing the proportion of our review catagorized by 'satisfied' and 'not satisfied' labels
```


```python
ax = df1['Satisfied'].value_counts().plot(kind='bar',
                                    figsize=(8,8),
                                    title="Sentiment of Customer Extraced from Restaurant's reviews")
ax.set_xlabel("Sentiment of Customer")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_76_0.png)
    


Because the review of cutomer comprise of 'satisfied review' more thatn 'not satisfied review' significantly, this could lead to 'Imbalanced of sentimental class',which might affect model to be biased. However, in this report, dealing with that issue is out of scope so we will randome pick samples from both group in the equal amount

# Get a sample set 

## Funtion to get sample set


```python
# function to get sample set from review of customer with the same amount
# the goal of doing thing because we want to eliminate the imbalancing of data set
def get_top_data(top_n = 20000):
    top_data_df_positive = df1[df1['Satisfied'] == 'satisfied'].head(top_n)
    top_data_df_negative = df1[df1['Satisfied'] == 'not satisfied'].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
    return top_data_df_small

```


```python
# extract 20,000 each
df2 = get_top_data(top_n=20000)
```


```python
ax = df2['Satisfied'].value_counts().plot(kind='bar',
                                    figsize=(8,8),
                                    title="Sentiment of Customer Extraced from Restaurant's reviews")
ax.set_xlabel("Sentiment of Customer")
ax.set_ylabel("Frequency")
plt.show()

#the problem of imbalancing data set is gone
```


    
![png](output_82_0.png)
    



```python
#Tokenization
#seperate text into single word and this will help when transforming text to numeric value

from gensim.utils import simple_preprocess
# Tokenize the text column to get the new column 'tokenized_text'
df2['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df2['text_clean']] 
print(df2['tokenized_text'].head(10))
```

    id
    0     [keep, cup, counter, well, made, sturdy, produ...
    1     [bar, pretty, good, taste, like, cinnamon, app...
    2     [love, pantry, cook, batch, rice, add, sauce, ...
    3     [used, another, brand, tonkotsu, flavor, noodl...
    4     [herr, favorite, chip, brand, fan, salsa, love...
    5     [absolutely, good, french, vanilla, cappuccino...
    6     [cheaper, chain, cup, make, home, stuff, aweso...
    8     [bought, coffee, amazon, special, promotion, g...
    9     [dog, love, zuke, treat, one, acceptation, lik...
    11    [cereal, like, chex, healthier, outstanding, f...
    Name: tokenized_text, dtype: object
    


```python
[col for col in df2]
```




    ['Time',
     'ProfileName',
     'Summary',
     'Text',
     'Score',
     'Satisfied',
     'text_clean',
     'tokenized_text']




```python
from gensim.parsing.porter import PorterStemmer
porter_stemmer = PorterStemmer()
# Get the stemmed_tokens
df2['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in df2['tokenized_text'] ]
df2['stemmed_tokens'].head(10)
```




    id
    0     [keep, cup, counter, well, made, sturdi, produ...
    1     [bar, pretti, good, tast, like, cinnamon, appl...
    2     [love, pantri, cook, batch, rice, add, sauc, d...
    3     [us, anoth, brand, tonkotsu, flavor, noodl, im...
    4     [herr, favorit, chip, brand, fan, salsa, love,...
    5     [absolut, good, french, vanilla, cappuccino, b...
    6     [cheaper, chain, cup, make, home, stuff, aweso...
    8     [bought, coffe, amazon, special, promot, go, e...
    9     [dog, love, zuke, treat, on, accept, like, muc...
    11    [cereal, like, chex, healthier, outstand, flav...
    Name: stemmed_tokens, dtype: object




```python
tmp =df2.iloc[::2000, [6, 7]]
with pd.option_context('display.max_colwidth', None):
  display(tmp)
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
      <th>text_clean</th>
      <th>tokenized_text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>keep cup counter well made sturdy product little stiff pull rather falling self used included screw make sure stay pleased</td>
      <td>[keep, cup, counter, well, made, sturdy, product, little, stiff, pull, rather, falling, self, used, included, screw, make, sure, stay, pleased]</td>
    </tr>
    <tr>
      <th>2572</th>
      <td>dog easy finding treat one fit bill two aussie shepherd get full piece min pin chihuahua mix get cut half love always look forward special treat</td>
      <td>[dog, easy, finding, treat, one, fit, bill, two, aussie, shepherd, get, full, piece, min, pin, chihuahua, mix, get, cut, half, love, always, look, forward, special, treat]</td>
    </tr>
    <tr>
      <th>5116</th>
      <td>reading review confused think anyone talking tea sounded like comment pertained prince peace green tea anything instant dong quai red date tea clicked said remarkable tea delicious instant wait steep really good love taste enjoy uniquely bitter flavor dong quai bitter yummy know dong quai considered ginseng woman beceause high vit b help keep woman becoming anemic due monthly cycle also said help regulate irregular period used daily basis said tea special help draw one energy downwards red date add effect red color st chakra word tea aphrodisiac quality create pleasnt feeling taken bed good taste good snap make quibbling</td>
      <td>[reading, review, confused, think, anyone, talking, tea, sounded, like, comment, pertained, prince, peace, green, tea, anything, instant, dong, quai, red, date, tea, clicked, said, remarkable, tea, delicious, instant, wait, steep, really, good, love, taste, enjoy, uniquely, bitter, flavor, dong, quai, bitter, yummy, know, dong, quai, considered, ginseng, woman, beceause, high, vit, help, keep, woman, becoming, anemic, due, monthly, cycle, also, said, help, regulate, irregular, period, used, daily, basis, said, tea, special, help, draw, one, energy, downwards, red, date, add, effect, red, color, st, chakra, word, tea, aphrodisiac, quality, create, pleasnt, feeling, taken, bed, good, taste, good, snap, make, quibbling]</td>
    </tr>
    <tr>
      <th>7676</th>
      <td>wow find avid latte drinker refuse pay outlandish price local coffee shop purchased machine couple year ago find supplier using flavor add coffee going business thankfully amazon com came rescue get convienence flavor delivered home paying le per bottle delivered thank amazon com loyal customer illinois</td>
      <td>[wow, find, avid, latte, drinker, refuse, pay, outlandish, price, local, coffee, shop, purchased, machine, couple, year, ago, find, supplier, using, flavor, add, coffee, going, business, thankfully, amazon, com, came, rescue, get, convienence, flavor, delivered, home, paying, le, per, bottle, delivered, thank, amazon, com, loyal, customer, illinois]</td>
    </tr>
    <tr>
      <th>10212</th>
      <td>mo old simply love stuff st official finger food think combo plus taste good dissolve easy mouth kind important teeth mom comment wish little green even close green color like waved green bowl making someone ought sell stuff mixed case go thru one container every day</td>
      <td>[mo, old, simply, love, stuff, st, official, finger, food, think, combo, plus, taste, good, dissolve, easy, mouth, kind, important, teeth, mom, comment, wish, little, green, even, close, green, color, like, waved, green, bowl, making, someone, ought, sell, stuff, mixed, case, go, thru, one, container, every, day]</td>
    </tr>
    <tr>
      <th>12784</th>
      <td>extremely better wilton buy never use nasty stuff actually edible unlike product easy work</td>
      <td>[extremely, better, wilton, buy, never, use, nasty, stuff, actually, edible, unlike, product, easy, work]</td>
    </tr>
    <tr>
      <th>15326</th>
      <td>using french market coffee many year moving guam store brought sent went mainland last year thrilled find could order amazon reasonable price automatic shipment coffee never bitter due chicory robust tasty use anything else</td>
      <td>[using, french, market, coffee, many, year, moving, guam, store, brought, sent, went, mainland, last, year, thrilled, find, could, order, amazon, reasonable, price, automatic, shipment, coffee, never, bitter, due, chicory, robust, tasty, use, anything, else]</td>
    </tr>
    <tr>
      <th>17900</th>
      <td>dog href http www amazon com gp product b j jkgo canidae dry dog food lamb meal brown rice formula pound bag year recently added diet stool firm seems like crazy taste texture little thick even mixed water</td>
      <td>[dog, href, http, www, amazon, com, gp, product, jkgo, canidae, dry, dog, food, lamb, meal, brown, rice, formula, pound, bag, year, recently, added, diet, stool, firm, seems, like, crazy, taste, texture, little, thick, even, mixed, water]</td>
    </tr>
    <tr>
      <th>20427</th>
      <td>santa cruz soft baked oatmeal raisin cookie one best ever flavor wonderful spice make think eating holiday pastry put plate cooky always first go call adult cookie child love price good delivered door could ask whole line cooky wonderful try find youself happy eating</td>
      <td>[santa, cruz, soft, baked, oatmeal, raisin, cookie, one, best, ever, flavor, wonderful, spice, make, think, eating, holiday, pastry, put, plate, cooky, always, first, go, call, adult, cookie, child, love, price, good, delivered, door, could, ask, whole, line, cooky, wonderful, try, find, youself, happy, eating]</td>
    </tr>
    <tr>
      <th>23008</th>
      <td>husband love tea drink antioxidant content difficulty finding favorite grocery store simply order amazon</td>
      <td>[husband, love, tea, drink, antioxidant, content, difficulty, finding, favorite, grocery, store, simply, order, amazon]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pop nice never get taste like movie theater popcorn even come close gave star popping taste</td>
      <td>[pop, nice, never, get, taste, like, movie, theater, popcorn, even, come, close, gave, star, popping, taste]</td>
    </tr>
    <tr>
      <th>9236</th>
      <td>href http www amazon com gp product b la vegetable base first bought product heb randalls sauce bought product get sauce like well used anyway good taste change buy would</td>
      <td>[href, http, www, amazon, com, gp, product, la, vegetable, base, first, bought, product, heb, randalls, sauce, bought, product, get, sauce, like, well, used, anyway, good, taste, change, buy, would]</td>
    </tr>
    <tr>
      <th>18415</th>
      <td>liquid v fish v flavor opened expecting large amount liquid ended spilling part table price expecting fish put another way fish swimming liquid flavor good give star could much better buy find appel brunswick</td>
      <td>[liquid, fish, flavor, opened, expecting, large, amount, liquid, ended, spilling, part, table, price, expecting, fish, put, another, way, fish, swimming, liquid, flavor, good, give, star, could, much, better, buy, find, appel, brunswick]</td>
    </tr>
    <tr>
      <th>27414</th>
      <td>tried one big bowl hot spicy almost identical taste mainly use main base soup okay first two time eat however burning taste get old quick known base used one would bought big bowl soup instead smaller bowl sized noodle</td>
      <td>[tried, one, big, bowl, hot, spicy, almost, identical, taste, mainly, use, main, base, soup, okay, first, two, time, eat, however, burning, taste, get, old, quick, known, base, used, one, would, bought, big, bowl, soup, instead, smaller, bowl, sized, noodle]</td>
    </tr>
    <tr>
      <th>36538</th>
      <td>love mcdougall food product one quite measure aftertaste enjoy find single container try first recommend</td>
      <td>[love, mcdougall, food, product, one, quite, measure, aftertaste, enjoy, find, single, container, try, first, recommend]</td>
    </tr>
    <tr>
      <th>45800</th>
      <td>get tea asian grocery around dollar tea good rip</td>
      <td>[get, tea, asian, grocery, around, dollar, tea, good, rip]</td>
    </tr>
    <tr>
      <th>55169</th>
      <td>one ate product one clean vomit product eaten offered new chew dog first cared le record love chew drug choice thought maybe reluctant try something new left chew little later ate seemed enjoy afterward threw twice sure problem chew buying dog certainly recommending anyone else thought giving remaining chew spca dog decided make poochies sick sadly throw rest away</td>
      <td>[one, ate, product, one, clean, vomit, product, eaten, offered, new, chew, dog, first, cared, le, record, love, chew, drug, choice, thought, maybe, reluctant, try, something, new, left, chew, little, later, ate, seemed, enjoy, afterward, threw, twice, sure, problem, chew, buying, dog, certainly, recommending, anyone, else, thought, giving, remaining, chew, spca, dog, decided, make, poochies, sick, sadly, throw, rest, away]</td>
    </tr>
    <tr>
      <th>64486</th>
      <td>incredibly embarrassed basket thought sending something substance based seller description picture cost sister law suffered incredibly life threatening illness received basket town family thinking something use offer guest local relative snack tea visited cheese basket nothing expensive cracker school lunch size packet chocolate chip cooky embarrassed many company offer le expensive basket greater good order basket barb dv</td>
      <td>[incredibly, embarrassed, basket, thought, sending, something, substance, based, seller, description, picture, cost, sister, law, suffered, incredibly, life, threatening, illness, received, basket, town, family, thinking, something, use, offer, guest, local, relative, snack, tea, visited, cheese, basket, nothing, expensive, cracker, school, lunch, size, packet, chocolate, chip, cooky, embarrassed, many, company, offer, le, expensive, basket, greater, good, order, basket, barb, dv]</td>
    </tr>
    <tr>
      <th>73778</th>
      <td>accurate description product ordered received one box ten bar bar great since one box worth buying</td>
      <td>[accurate, description, product, ordered, received, one, box, ten, bar, bar, great, since, one, box, worth, buying]</td>
    </tr>
    <tr>
      <th>82541</th>
      <td>initial review one star pro amazon delivery ordered yesterday prime membership ontrac delivery placed doorstep today saturday thank amazon con either coffee bad make work tried approach senseo machine almost weight two senseo pod unfortunately longer sold amazon used two pod holder fit one pod holder coffee appeared brew e water ran machine fine result even close passable way tried result senseo user understand term right brew button two bar level far weak right brew button one bar level weak left brew button one bar level strong quite bitter left brew button two bar level weak bitter moral story order pod unless specifically designed machine since amazon accept return food product least let loss stand lesson others avoid senseo machine fit suspect coffee good message amazon sold coffee machine sell senseo brand pod anymore something specifically fit g pod update revision one day later okay since stuck thing figured give easily went back senseo pod managed make reasonably good star coffee lesson learned important note senseo user vi vi assume g pod make sure pod oriented correct side use right brew button one bar level yield tasty cup coffee certainly two cup yield know cost benefit v using gram pod reality bottom line revised star moral story give</td>
      <td>[initial, review, one, star, pro, amazon, delivery, ordered, yesterday, prime, membership, ontrac, delivery, placed, doorstep, today, saturday, thank, amazon, con, either, coffee, bad, make, work, tried, approach, senseo, machine, almost, weight, two, senseo, pod, unfortunately, longer, sold, amazon, used, two, pod, holder, fit, one, pod, holder, coffee, appeared, brew, water, ran, machine, fine, result, even, close, passable, way, tried, result, senseo, user, understand, term, right, brew, button, two, bar, level, far, weak, right, brew, button, one, bar, level, weak, left, brew, button, one, bar, level, strong, quite, bitter, left, brew, button, two, bar, level, weak, bitter, moral, story, order, pod, ...]</td>
    </tr>
  </tbody>
</table>
</div>


#Splitting into Train and Test Sets:
Train data would be used to train the model and test data is the data on which the model would predict the classes and it will be compared with original labels to check the accuracy or other model test metrics.

NOTE: In this case I will split data into 70:30


```python
df2
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
      <th>Time</th>
      <th>ProfileName</th>
      <th>Summary</th>
      <th>Text</th>
      <th>Score</th>
      <th>Satisfied</th>
      <th>text_clean</th>
      <th>tokenized_text</th>
      <th>stemmed_tokens</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-02-11 08:00:00</td>
      <td>cac Idaho</td>
      <td>Great for small kitchens.</td>
      <td>Keeps the cups off the counter!! It is a very ...</td>
      <td>5</td>
      <td>satisfied</td>
      <td>keep cup counter well made sturdy product litt...</td>
      <td>[keep, cup, counter, well, made, sturdy, produ...</td>
      <td>[keep, cup, counter, well, made, sturdi, produ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-07-24 08:00:00</td>
      <td>Fielden A. Coleman "Coleblooded1"</td>
      <td>Good Taste!! Good Price too!!</td>
      <td>The bar is pretty good. Taste more like cinnam...</td>
      <td>4</td>
      <td>satisfied</td>
      <td>bar pretty good taste like cinnamon apple pie ...</td>
      <td>[bar, pretty, good, taste, like, cinnamon, app...</td>
      <td>[bar, pretti, good, tast, like, cinnamon, appl...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-10-16 08:00:00</td>
      <td>Auskan "Auskan"</td>
      <td>Easy &amp; delicious</td>
      <td>I love having this in my pantry.  I cook a bat...</td>
      <td>5</td>
      <td>satisfied</td>
      <td>love pantry cook batch rice add sauce dinner s...</td>
      <td>[love, pantry, cook, batch, rice, add, sauce, ...</td>
      <td>[love, pantri, cook, batch, rice, add, sauc, d...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-08-24 08:00:00</td>
      <td>chicago</td>
      <td>This is the best!</td>
      <td>We used to have another brand tonkotsu flavor ...</td>
      <td>5</td>
      <td>satisfied</td>
      <td>used another brand tonkotsu flavor noodle impo...</td>
      <td>[used, another, brand, tonkotsu, flavor, noodl...</td>
      <td>[us, anoth, brand, tonkotsu, flavor, noodl, im...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-07-13 08:00:00</td>
      <td>you suckkk</td>
      <td>Yum</td>
      <td>Herr's are my favorite chip brand. I am not su...</td>
      <td>5</td>
      <td>satisfied</td>
      <td>herr favorite chip brand fan salsa love chip</td>
      <td>[herr, favorite, chip, brand, fan, salsa, love...</td>
      <td>[herr, favorit, chip, brand, fan, salsa, love,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91440</th>
      <td>2012-09-05 08:00:00</td>
      <td>JFMile</td>
      <td>Son wasn't a fan</td>
      <td>My son simply did not like this flavor jar.  H...</td>
      <td>3</td>
      <td>not satisfied</td>
      <td>son simply like flavor jar like pea spinach ev...</td>
      <td>[son, simply, like, flavor, jar, like, pea, sp...</td>
      <td>[son, simpli, like, flavor, jar, like, pea, sp...</td>
    </tr>
    <tr>
      <th>91449</th>
      <td>2011-05-03 08:00:00</td>
      <td>J. S. Bowen</td>
      <td>THEY HAVE CHANGED THIS TEA</td>
      <td>I USED TO LOVE THIS TEA WHEN IT WAS CALLED "WH...</td>
      <td>1</td>
      <td>not satisfied</td>
      <td>used love tea called white tea made peony plai...</td>
      <td>[used, love, tea, called, white, tea, made, pe...</td>
      <td>[us, love, tea, call, white, tea, made, peoni,...</td>
    </tr>
    <tr>
      <th>91451</th>
      <td>2010-05-21 08:00:00</td>
      <td>Jennifer Hines "Jen H"</td>
      <td>Great Cocoa, Priced Too High</td>
      <td>While I love the taste of Green Mountain Hot C...</td>
      <td>2</td>
      <td>not satisfied</td>
      <td>love taste green mountain hot cocoa price k cu...</td>
      <td>[love, taste, green, mountain, hot, cocoa, pri...</td>
      <td>[love, tast, green, mountain, hot, cocoa, pric...</td>
    </tr>
    <tr>
      <th>91454</th>
      <td>2011-10-04 08:00:00</td>
      <td>B. McMahon</td>
      <td>Yuck</td>
      <td>I didn't like this brand of coconut water it h...</td>
      <td>2</td>
      <td>not satisfied</td>
      <td>like brand coconut water strange taste brand l...</td>
      <td>[like, brand, coconut, water, strange, taste, ...</td>
      <td>[like, brand, coconut, water, strang, tast, br...</td>
    </tr>
    <tr>
      <th>91455</th>
      <td>2012-10-10 08:00:00</td>
      <td>Burnadette Cerda</td>
      <td>Not as good as I thought it would be.</td>
      <td>The seller was amazing and fast so I would ord...</td>
      <td>1</td>
      <td>not satisfied</td>
      <td>seller amazing fast would order sad say tea ex...</td>
      <td>[seller, amazing, fast, would, order, sad, say...</td>
      <td>[seller, amaz, fast, would, order, sad, sai, t...</td>
    </tr>
  </tbody>
</table>
<p>40000 rows × 9 columns</p>
</div>



# Split train_test set


```python
# a function to split data into traing set and testing set with summary 

from sklearn.model_selection import train_test_split
# Train Test Split Function
def split_train_test(df2, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(df2[['stemmed_tokens']], 
                                                        df2['Satisfied'], 
                                                        shuffle=shuffle_state,
                                                        test_size=test_size, 
                                                        random_state=15)
    print("Value counts for Train sentiment")
    print(Y_train.value_counts())
    print('\n')
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print('\n')
    print(type(X_train))
    print(type(Y_train))
    print('\n')
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test
    
```


```python
X_train, X_test, Y_train, Y_test = split_train_test(df2)
```

    Value counts for Train sentiment
    satisfied        14027
    not satisfied    13973
    Name: Satisfied, dtype: int64
    
    
    Value counts for Test sentiments
    not satisfied    6027
    satisfied        5973
    Name: Satisfied, dtype: int64
    
    
    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.series.Series'>
    
    
          id                                     stemmed_tokens
    0  50319  [pleas, try, avoid, disappoint, wai, pack, qua...
    1  58934  [deliveri, quick, easi, howev, product, best, ...
    2  30562  [order, brother, law, like, coffe, ship, slow,...
    3   6542  [on, best, chocol, bar, tast, recommend, frien...
    4  55060  [want, like, coffe, like, bui, came, huge, bag...
    

x_train: The training part of the first sequence (x)
x_test: The test part of the first sequence (x)
y_train: The training part of the second sequence (y)
y_test: The test part of the second sequence (y)

More Detail: splitting training ans testing set
https://realpython.com/train-test-split-python-data/#:~:text=x_train%20%3A%20The%20training%20part%20of,of%20the%20second%20sequence%20(%20y%20)


```python
X_train
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
      <th>id</th>
      <th>stemmed_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50319</td>
      <td>[pleas, try, avoid, disappoint, wai, pack, qua...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58934</td>
      <td>[deliveri, quick, easi, howev, product, best, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30562</td>
      <td>[order, brother, law, like, coffe, ship, slow,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6542</td>
      <td>[on, best, chocol, bar, tast, recommend, frien...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55060</td>
      <td>[want, like, coffe, like, bui, came, huge, bag...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>71548</td>
      <td>[found, dry, stale, also, rel, high, calori, p...</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>88314</td>
      <td>[greet, blew, first, request, review, howev, r...</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>3467</td>
      <td>[fantast, realli, le, calori, fat, eat, spoon,...</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>10307</td>
      <td>[number, cat, medic, problem, tri, number, wai...</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>9732</td>
      <td>[us, head, shoulder, moder, dandruff, work, we...</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```python
X_test
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
      <th>id</th>
      <th>stemmed_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80957</td>
      <td>[though, show, differ, flavor, bag, realli, at...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90660</td>
      <td>[follow, review, larger, size, product, offer,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79126</td>
      <td>[watch, advertis, tofu, noodl, decid, try, sup...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17701</td>
      <td>[kid, seriou, allergi, tri, bake, muffin, us, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8261</td>
      <td>[on, pack, pack, price, care, mistak, two, pac...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11995</th>
      <td>2788</td>
      <td>[great, choic, like, cinnamon, roll, flavor, c...</td>
    </tr>
    <tr>
      <th>11996</th>
      <td>13761</td>
      <td>[get, cake, mix, auto, deliveri, long, rememb,...</td>
    </tr>
    <tr>
      <th>11997</th>
      <td>20579</td>
      <td>[barri, farm, establish, oct, bill, linda, bar...</td>
    </tr>
    <tr>
      <th>11998</th>
      <td>19695</td>
      <td>[love, herbal, tea, delici, tast, like, oolong...</td>
    </tr>
    <tr>
      <th>11999</th>
      <td>17727</td>
      <td>[dog, alwai, consum, love, pedigre, ag, need, ...</td>
    </tr>
  </tbody>
</table>
<p>12000 rows × 2 columns</p>
</div>




```python
Y_train
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
      <th>id</th>
      <th>Satisfied</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50319</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58934</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30562</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6542</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55060</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>71548</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>88314</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>3467</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>10307</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>9732</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```python
Y_test
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
      <th>id</th>
      <th>Satisfied</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80957</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90660</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79126</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17701</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8261</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11995</th>
      <td>2788</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>11996</th>
      <td>13761</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>11997</th>
      <td>20579</td>
      <td>not satisfied</td>
    </tr>
    <tr>
      <th>11998</th>
      <td>19695</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>11999</th>
      <td>17727</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
<p>12000 rows × 2 columns</p>
</div>



# Word2Vec

Feature Extraction
we will use Word2Vec Model, which is a pre-trained model to fitting 


```python
from gensim.models import Word2Vec
import time
# Skip-gram model (sg = 1)
vector_size=1000
window = 5
min_count = 1
workers = 3
sg = 1

word2vec_model_file =  'word2vec_' + str(vector_size) + '.model'
start_time = time.time()
stemmed_tokens = pd.Series(df2['stemmed_tokens']).values
# Train the Word2Vec Model
w2v_model = Word2Vec(stemmed_tokens, min_count = min_count,vector_size=vector_size ,workers = workers, window = window, sg = sg)
print("Time taken to train word2vec model: " + str(time.time() - start_time))

# Because this process might take a long time as well, so i save the file 'word2vec_model_file'
w2v_model.save(word2vec_model_file)
```

    Time taken to train word2vec model: 67.84997940063477
    


```python
# after fitting the model each word in our review can be perceived as vectors
#and now, it being able to find some kind of correlation between those words

# Load the model from the model file
w2v_model = Word2Vec.load(word2vec_model_file)

# Most Similar word
print(w2v_model.wv.most_similar('well'))

#Now the model know that some words  which have the similar meaning would be represented by corresponding value 
w2v_model.wv.similarity('good', 'worthwhil')

```

    [('gosh', 0.595484733581543), ('strictli', 0.5866800546646118), ('reg', 0.5853108763694763), ('lacklust', 0.5839182138442993), ('testament', 0.5802475214004517), ('lastli', 0.5793293714523315), ('swap', 0.5771440267562866), ('safest', 0.5766143202781677), ('definitli', 0.5758413076400757), ('wallet', 0.5745723247528076)]
    




    0.70346117




```python
# The model can classify the word that not belong to a group
w2v_model.wv.doesnt_match(['good', 'charm', 'amazingli','bad','well']) 
```




    'bad'




```python
w2v_model.wv.similarity('good', 'worthwhil')
```




    0.70346117




```python
w2v_model.wv.similarity('bad', 'bitter')
```




    0.3782112




```python
w2v_model.wv.similarity('bad', 'good') # need to fix this
```




    0.55581874




```python
w2v_model.wv.most_similar(positive="bad")
```




    [('terribl', 0.6352477669715881),
     ('weird', 0.6308624744415283),
     ('wierd', 0.6243007183074951),
     ('keen', 0.6209333539009094),
     ('becuas', 0.6188809275627136),
     ('that', 0.6155038475990295),
     ('interestingli', 0.6136985421180725),
     ('horrid', 0.6136499643325806),
     ('bum', 0.6124559640884399),
     ('yucki', 0.6093966364860535)]




```python
w2v_model.wv.most_similar(positive="chip")
```




    [('popchip', 0.6778391003608704),
     ('kettl', 0.6556416153907776),
     ('ahoi', 0.6520850658416748),
     ('frito', 0.6359274983406067),
     ('tortilla', 0.6302423477172852),
     ('pringl', 0.6267793774604797),
     ('terra', 0.6198611855506897),
     ('mesquit', 0.6150088310241699),
     ('ruffl', 0.6042578816413879),
     ('pretzel', 0.602069616317749)]



# Core Process of Word2Vec

From now, we need will work with traing set to fitting the model before making prediction.
we loop through X_train and X_test, which previously splitted beforehand and we kind of find the mean of each vector in a reviewand used that as a representative of tone in that review


```python
#for training set
# we find the mean of vector in each review and used that as a representative of tone in that review
#write them into csv file.
word2vec_filename = 'train_review_word2vec.csv'
with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in X_train.iterrows():
        model_vector = (np.mean([w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(1000))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:  
            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
        else:
            line1 = ",".join([str(0) for i in range(1000)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')
```

    C:\Python\lib\site-packages\numpy\core\fromnumeric.py:3440: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    C:\Python\lib\site-packages\numpy\core\_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    


```python
#for teseting set
#find the mean
#Also,write them into csv file.
word2vec_filename = 'test_review_word2vec.csv'
with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in X_test.iterrows(): #itterows(); used to loop over each review annd find the mean to represent each review
        model_vector = (np.mean([w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(1000))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:  
            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
        else:
            line1 = ",".join([str(0) for i in range(1000)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')
```


```python
#now we need algorithm for prediction, in this case,  I use RandomForestClassifier
import time
#import RandomForestClassifier, this is the algorithm that will be used for classification
from sklearn.ensemble import RandomForestClassifier

# Load from the filename
trainvec = pd.read_csv('train_review_word2vec.csv') # training
testvec = pd.read_csv('test_review_word2vec.csv') # testing

#Initialize the model
forest_word2vec = RandomForestClassifier(n_estimators = 100)

start_time = time.time()
# Fit the model
forest_word2vec.fit(trainvec, Y_train['Satisfied']) # fitting the model; find the coefficients or the model
print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))
```

    Time taken to fit the model with word2vec vectors: 80.12357902526855
    


```python
#use model that being fitted already to predict the result
# the result is either the review in testset is 'satisfied', or 'not satisfied
result = forest_word2vec.predict(testvec) 
```


```python
result.shape
```




    (12000,)




```python
result[::10]
```




    array(['not satisfied', 'satisfied', 'satisfied', ..., 'satisfied',
           'satisfied', 'satisfied'], dtype=object)




```python
#append the result to our test set
Y_test['Predict'] = result
```


```python
Y_test['review'] = X_test['stemmed_tokens']
```


```python
# the end result
Y_test[::500]
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
      <th>id</th>
      <th>Satisfied</th>
      <th>Predict</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80957</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[though, show, differ, flavor, bag, realli, at...</td>
    </tr>
    <tr>
      <th>500</th>
      <td>80996</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[whenth, packag, arriv, two, can, open, empti,...</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>11473</td>
      <td>satisfied</td>
      <td>not satisfied</td>
      <td>[crunchi, cooki, creami, center, cooki, part, ...</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>43875</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[complet, underwhelm, overpr, would, consid, t...</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>20155</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[year, old, finicki, cat, absolut, love, food,...</td>
    </tr>
    <tr>
      <th>2500</th>
      <td>15199</td>
      <td>satisfied</td>
      <td>not satisfied</td>
      <td>[oh, boi, dubbl, bubbl, bubbl, gum, giant, ind...</td>
    </tr>
    <tr>
      <th>3000</th>
      <td>14302</td>
      <td>satisfied</td>
      <td>not satisfied</td>
      <td>[good, appl, soft, hard, flavor, good, packag,...</td>
    </tr>
    <tr>
      <th>3500</th>
      <td>7918</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[cook, grain, soft, creami, tast, somewhat, ri...</td>
    </tr>
    <tr>
      <th>4000</th>
      <td>91399</td>
      <td>not satisfied</td>
      <td>satisfied</td>
      <td>[four, rescu, cat, give, iam, said, organ, tho...</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>36933</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[mind, pai, bag, dog, love, expens, look, bag,...</td>
    </tr>
    <tr>
      <th>5000</th>
      <td>2063</td>
      <td>satisfied</td>
      <td>not satisfied</td>
      <td>[struggl, dry, skin, excit, try, product, dove...</td>
    </tr>
    <tr>
      <th>5500</th>
      <td>24499</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[love, almond, flour, graini, us, bake, fine, ...</td>
    </tr>
    <tr>
      <th>6000</th>
      <td>20320</td>
      <td>not satisfied</td>
      <td>satisfied</td>
      <td>[want, try, someth, healthi, differ, meusli, e...</td>
    </tr>
    <tr>
      <th>6500</th>
      <td>19246</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[italian, greyhound, love, treat, yet, notic, ...</td>
    </tr>
    <tr>
      <th>7000</th>
      <td>14868</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[wife, absolut, ador, cocoa, easi, make, hot, ...</td>
    </tr>
    <tr>
      <th>7500</th>
      <td>18741</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[alwai, love, walker, shortbread, shape, size,...</td>
    </tr>
    <tr>
      <th>8000</th>
      <td>49797</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[month, old, practic, live, cheerio, plum, org...</td>
    </tr>
    <tr>
      <th>8500</th>
      <td>67583</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[enjoi, dip, dress, compani, dip, strong, bitt...</td>
    </tr>
    <tr>
      <th>9000</th>
      <td>37978</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[us, yogi, tea, lemon, ginger, tea, chang, wan...</td>
    </tr>
    <tr>
      <th>9500</th>
      <td>22370</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[done, recur, shipment, product, steal, carb, ...</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>241</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[tulli, hous, blend, becom, coffe, alwai, hand...</td>
    </tr>
    <tr>
      <th>10500</th>
      <td>54662</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[like, type, coffe, surpris, blend, suit, tast...</td>
    </tr>
    <tr>
      <th>11000</th>
      <td>73658</td>
      <td>not satisfied</td>
      <td>not satisfied</td>
      <td>[purchas, option, cat, sinc, eat, chicken, muc...</td>
    </tr>
    <tr>
      <th>11500</th>
      <td>352</td>
      <td>satisfied</td>
      <td>satisfied</td>
      <td>[love, coffe, wonder, flavor, take, litll, mak...</td>
    </tr>
  </tbody>
</table>
</div>



# Evaluate the model


```python
#using a funciton to evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test['Satisfied'],result, zero_division=0))
```

                   precision    recall  f1-score   support
    
    not satisfied       0.80      0.80      0.80      6027
        satisfied       0.79      0.80      0.80      5973
    
         accuracy                           0.80     12000
        macro avg       0.80      0.80      0.80     12000
     weighted avg       0.80      0.80      0.80     12000
    
    

# Confusion Matrix


```python
#visualizing the evaluation of model with heatmap
```


```python
# we will use confusion matrix and feed that into heatmap
cf_matrix = confusion_matrix(Y_test['Satisfied'], result)
```


```python
cf_matrix
```




    array([[4792, 1235],
           [1207, 4766]], dtype=int64)




```python
#visualizing the evaluation of model with heatmap
import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='YlGn', fmt='.1f')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()
```


    
![png](output_124_0.png)
    


**NOTE**
x_train: The training part of the first sequence (x)
x_test: The test part of the first sequence (x)
y_train: The training part of the second sequence (y)
y_test: The test part of the second sequence (y)
