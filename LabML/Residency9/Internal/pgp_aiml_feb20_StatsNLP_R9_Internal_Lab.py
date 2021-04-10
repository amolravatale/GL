#!/usr/bin/env python
# coding: utf-8

# <img src="http://drive.google.com/uc?export=view&id=1tpOCamr9aWz817atPnyXus8w5gJ3mIts" width=500px>
# 
# Proprietary content. © Great Learning. All Rights Reserved. Unauthorized use or distribution prohibited.

# # Mobile Phone Review Analysis

# ## Context
# 
# The product companies can utilize the detailed review comments to gather insights from the end user. Most of the products are sold via e-commerce sites like Flipkart or Amazon where customers can buy a product and give their review about the product on the web site. 
# Product managers can identify the relevant reviews from the website and run a sentiment analysis tool to understand what the sentiments of the customer are. Based on their sentiments, they can identify what users think of the current product. Are they happy? Discontent? 
# They can also come up with a document that lists the features, the team needs to focus on for making the product better. 
# 
# ## Objective
# 
# Given the review data rating label, we will try to get insights about various brands and their ratings using text analytics and build a model to predict the rating and overall sentiment. 
# 

# ### Package version
# 
# - tensorflow==2.3.0
# - scikit-learn==0.22.2.post1
# - pandas==1.0.5
# - numpy==1.18.5
# - matplotlib==3.2.2
# - google==2.0.3

# ### Data Dictionary 
# 
# product_data.csv - contains product details
# - 'asin',  - Product ASIN
# - 'brand', - Product Brand
# - 'title', - Product Title
# - 'url',  - Product URL
# - 'image', - Product Image URL
# - 'rating',- Product Avg. Rating
# - 'reviewUrl' - Product Review Page URL
# - 'totalReviews' - Product Total Reviews
# - ‘price’ - Product Price ($)
# - ‘originalPrice’ - Product Original Price ($)
#  
# reviews.csv  - contains user review details
#  
# - 'asin' - Product ASIN
# - 'name' - Reviewer Name
# - 'rating' - Reviewer Rating (scale 1 to 5)
# - 'date'  - Review Date
# - 'verified' - Valid Customer
# - 'title'  - Review Title
# - 'body'  - Review Content
# - 'helpfulVotes  - Helpful Feedbacks
# 

# ## Table of Content
# 
# 1. Import Libraries
# 
# 2. Setting options
# 
# 3. Read Data
# 
# 4. Data Analysis and EDA
# 
# 5. Text preprocessing and Vectorization
# 
# 6. Model building
# 
# 7. Conclusion and Interpretation

# ## 1. Import Libraries

# Let us start by mounting the drive

# In[2]:


#connect to google drive
#from google.colab import drive
#import os
#drive.mount('/content/drive')


# In[3]:


#os.chdir('/content/drive/MyDrive/LablFiles/Stat_NLP_Internal_Lab/')


# Let us check for the version of installed tensorflow.

# In[4]:

# used to supress display of warnings
import warnings

# Pandas is used for data manipulation and analysis
import pandas as pd
import matplotlib.pyplot as plt


# os is used to provide a way of using operating system dependent functionality
# We use it for setting working folder

# Numpy is used for large, multi-dimensional arrays and matrices, along with mathematical operators on these arrays

# Matplotlib is a data visualization library for 2D plots of arrays, built on NumPy arrays 
# and designed to work with the broader SciPy stack

#get_ipython().run_line_magic('matplotlib', 'inline')


# Seaborn is based on matplotlib, which aids in drawing attractive and informative statistical graphics.
#import tensorflow
#print(tensorflow.__version__)


# ## 2. Setting Options

# In[18]:


# suppress display of warnings
warnings.filterwarnings('ignore')

# display all dataframe columns
pd.options.display.max_columns = None

# to set the limit to 3 decimals
pd.options.display.float_format = '{:.7f}'.format

# display all dataframe rows
pd.options.display.max_rows = None


# ## 3. Read Data

# ### 3.1 Read the provided CSVs and check 5 random samples and shape to understand the datasets

# In[39]:


productds = pd.read_csv('product_data.csv')

# In[40]:


productds.sample(5)


# In[41]:


productds.shape


# In[42]:


len(pd.unique(productds['asin']))


# Inferences
# 1) There are unique 720 mobile products

# In[43]:


reviewsds = pd.read_csv('reviews.csv')


# In[44]:


reviewsds.sample(5)


# In[37]:


reviewsds.shape


# ## 4.  Data Analysis and EDA

# ### 4.1 Drop unnecessary columns like 'url', 'image' from the product_data

# In[45]:


productds.columns


# In[46]:


productds.drop(['url','image','reviewUrl'],axis=1,inplace=True)


# In[47]:


productds.columns


# ### 4.2 Check statistical summary of both datasets. Note:- Include both numerical and object type columns.

# In[48]:


productds.describe(include='all').transpose()


# In[50]:


reviewsds.describe(include='all').transpose()


# In[51]:


productds['brand'].value_counts()


# In[53]:


reviewsds['rating'].value_counts()


# ### 4.3 From the above statistical summary, write inferences like count of unique products, top brand, top title, range of rating, price range, etc

# Inferences...
# 1) There are 720 unique mobile products
# 2) Samsung has 346 products which is 48%...so samsung has wide variety of mobile products and is top brand
# 3) Price Range and related aspects:
#    a) 50% of the products have price less than 199 
#    b) 75% of the products have price less than 336
#   The mean value of price is 235 and max is 999 
#   Thus the price distribution exhibits positive skewness

# ### 4.4 Analyze the distribution of ratings and other categorical features like brand, etc

# In[55]:


reviewsds['rating'].value_counts().plot(kind='bar')


# In[ ]:


productds['brand'].value_counts().plot(kind='bar')


# Inferences...
# 1) Almost 48% of the products are Samsung
# 2) more than 50% of the reviews are for Samsung products

# ### 4.5 Display average rating per brand

# In[56]:


productds.groupby('brand').mean()['rating']


# In[57]:


productds.groupby('brand').mean()['rating'].sort_values().plot(kind='bar')


# Inferences...
# 1) 'Xiaomi' brand has highest average rating.
# 2) 'Nokia' brand has lowest average rating.

# ### 4.6 Display average price per brand

# In[58]:


productds.groupby('brand').mean()['price']


# In[59]:


productds.groupby('brand').mean()['price'].sort_values().plot(kind='bar')


# In[60]:


productds['price'].describe()


# ### 4.7 Display average 'totalReviews' per brand

# In[61]:


productds.groupby('brand').mean()['totalReviews'].sort_values().plot(kind='bar')


# ### 4.8 Merge two datasets using 'asin' and check the shape of the final dataset

# In[63]:


dfprodreview= pd.merge(reviewsds,productds,on='asin',how='inner')


# In[64]:


dfprodreview.shape


# In[65]:


dfprodreview.sample(2)


# ### 4.9 Rename important features with appropriate names.
# Imortant features - "rating_x": "user_rating", "title_x": "review_title", "title_y": "item_title", "rating_y": "overall_rating"

# In[66]:


dfprodreview.rename(columns={"rating_x": "user_rating", "title_x": "review_title", "title_y": "item_title", "rating_y": "overall_rating"},inplace=True)


# In[67]:


dfprodreview.columns


# ### 4.10 Select rows having verified reviews and check the shape of the final dataset

# In[68]:


dfprodreview['verified'].value_counts()


# In[69]:


dfprodreview[dfprodreview['verified']==True].shape


# In[70]:


df_varified = dfprodreview[dfprodreview['verified']==True]


# In[71]:


df_varified.shape


# ### 4.11 Check the number of reviews for various brands and report the brand that have highest number of reviews

# In[72]:


df_varified['brand'].value_counts()


# In[73]:


df_varified['brand'].value_counts().plot(kind='bar')


# ### 4.12 Drop irrelevant columns and keep important features like 'brand','body','price','user_rating','review_title' for further analysis

# In[74]:


dfprodreview.columns


# In[75]:


finaldf = dfprodreview[['brand','body','price','user_rating','review_title']]


# In[76]:


finaldf.sample(5)


# ### 4.13 Perform univariate analysis. Check distribution of price, user_rating

# In[77]:


finaldf['user_rating'].value_counts().plot(kind='bar')


# ### 4.14 Create a new column called "sentiment". It should have value as 1 (positive) if the user_Rating is greater than 3, value as 0 (neutral) if the user_Rating == 3, and -1 (negative) is the user_Rating is less than 3.

# In[78]:


def sentiment(rating) :
  if rating == 1 or rating ==2 :  #Lets consider these values as negative sentiment
    return -1
  if rating == 4 or rating ==5 :  #Lets consider these values as positive sentiment
    return 1
  if rating == 3 :                #Lets consider this as neutral sentiment
    return 0
  


# In[80]:


finaldf['Sentiment'] = finaldf['user_rating'].apply(lambda x : sentiment(x))


# ### 4.15 Check frequency distribution of the 'sentiment'

# In[81]:


finaldf['Sentiment'].value_counts().plot(kind='bar')


# ### 4.16 Perform bivariate analysis. Check correlation/crosstab between features and write your inferences.

# In[82]:


pd.crosstab(finaldf['Sentiment'], finaldf['brand']).T


# In[83]:


pd.crosstab(finaldf['Sentiment'], finaldf['brand']).T.plot(kind='bar')


# In[84]:


pd.crosstab(finaldf['user_rating'], finaldf['brand']).T.plot(kind='bar')


# Based on above charts it seems that people aggresively rate their most postive or mmost negative experiance.
# The users have given rating in case of user rating 5 or 1 for all brands 

# ## 5. Text Preprocessing and Vectorization

# We will analyze the 'body' and 'review_title' to gain more understanding.
# 
# We will ppeform the below tasks
# 
# - Convert the text into lowercase
# - Remove punctuation
# - Remove stopwords (English, from nltk corpus)
# - Remove other keywords like "phone" and brand name

# ### 5.1 Change the datatype of the 'body' column to 'str' and convert it into lowercase. Print any two samples and check the output.

# In[85]:


finaldf.dtypes


# In[86]:


finaldf['body'] = finaldf['body'].astype('str')


# In[87]:


finaldf['body'] = finaldf['body'].str.lower()


# In[88]:


finaldf.sample(2)


# ### 5.2 Remove punctuations from the lowercased 'body' column and display at least two samples.

# In[89]:


finaldf['body'] = finaldf['body'].str.replace('[^\w\s]',' ')  #Lets replace all except words and space characters


# In[90]:


finaldf.sample(2)


# ### 5.3 Remove stop words from the above pre-processed 'body' column and display at least two samples.

# In[91]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')


# In[92]:


finaldf['body'] = finaldf['body'].apply(lambda x : ' '.join(word for word in x.split() if word not in (stop_words)))


# In[93]:


finaldf.sample(2)


# ### 5.4 Apply lemmatisation on the above preprocessed text and display a few samples

# In[94]:


import nltk 
nltk.download('wordnet') 
w_tokenizer = nltk.tokenize.WhitespaceTokenizer() 
lemmatizer = nltk.stem.WordNetLemmatizer() 


# In[95]:


def lemmatize_text_nltk(text) : 
  return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)] 


# In[96]:


finaldf['body_1']  = finaldf['body'].apply(lambda x : lemmatize_text_nltk(x))


# In[97]:


finaldf.sample(10)


# Seems that nltk library is unable to apply lemmatisation correctly for all reviews as I can still see workds with 'ing' 'ed'
# Lets try another library

# In[98]:


import spacy
nlp = spacy.load('en')


# In[99]:


def lemmatize_text_spacy(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# In[100]:


finaldf['body_2']  = finaldf['body'].apply(lambda x : lemmatize_text_spacy(x))


# In[101]:


finaldf.sample(10)


# Based on random manual observation of result of lemmatisation in 'body_1' (using nltk library) and 'body_2' (using space library).
# Space library seems to be giving better results. Lets use 'body_2' column for feature building

# ### 5.5 Write a function to check word frequency of the text

# In[102]:


def workFrequency(text) :
  lst = text.split()
  return len(lst)


# In[103]:


finaldf['WordFrequency'] = finaldf['body'].apply(lambda x : workFrequency(x))


# In[104]:


finaldf.sample(10)


# ### 5.6 Check word frequency of review having top ratings (sentiment = 1) and display output of few samples

# In[115]:


#Lets first filter and create dataset with reviews having only top rating i.e. user_rating =5 and user ratings =4
topratingds = finaldf[(finaldf['user_rating']==5) | (finaldf['user_rating']==4)]


# In[116]:


topratingds.sample(10)


# ### 5.7 Initialize tf-idf vectorizer and transform the preprocessed body text

# In[117]:


from sklearn.feature_extraction.text  import TfidfVectorizer


# In[118]:


tfidfvector = TfidfVectorizer()


# In[119]:


X = tfidfvector.fit_transform(finaldf['body_2'])


# ### 5.8 Segregate the data into dependent (sentiment) and independent (transformed body using tf-idf) features for building a classifier. 

# In[120]:


y = finaldf['Sentiment']


# ### 5.9 Split the data into Train & Test Sets

# In[121]:


from sklearn.model_selection import train_test_split


# In[122]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# ## 6. Model building

# ### 6.1 Build a random forest classifier to predict the 'sentiment'
# ### 6.2 Predict on test set
# ### 6.3 Check accuracy and confusion matrix

# In[123]:


from sklearn.ensemble import RandomForestClassifier


# In[124]:


model = RandomForestClassifier(n_estimators=150)
model.fit(x_train,y_train)


# In[125]:


model.score(x_train,y_train)


# In[126]:


model.score(x_test,y_test)


# In[129]:


X.shape


# In[128]:


x_train.shape


# In[130]:


y_test.shape


# ## 7. Write your conclusion

# In[ ]:

"""
1) The training accuracy is quite high...almost reaching 100 %. However test accuracy is 84 %
2) There could be multiple reasons for this difference in training and test accuracy as below
  a) The number of features are quite huge i.e. 30593
  b) The lemmatisation is not as efficient as we expect. We tried nltk and spacy libraries. Though space is more effiecnt 
     than nltk, it is still unanle to lemmatise many words. Due to this number of features could not be reduced
3) On one side we should see how to can reduce number of features and other side should try models on other techniques like dense layer, CNN etc.
"""
