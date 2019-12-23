#import libraries

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing

#loading data set

data = pd.read_csv('dataset.csv', error_bad_lines = False)

#check data set

data.head()
print('length of dataset reviews is ',len(data))
data.columns

#select only two columns necessary for sentiment analysis

data=data[['reviews.text','reviews.rating']]
data.head()
data.columns
data.dtypes

#drop null values

data[data.isnull().any(axis=1)]
np.sum(data.isnull().any(axis=1))
data=data.dropna(axis = 0, how ='any')
data.head()
np.sum(data.isnull().any(axis=1))
print(len(data))

#get information on data

data.info()

#visualize the wordcloud

neg = data['reviews.text']
neg_string = []
for t in neg:
    neg_string.append(t)
neg_string = pandas.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#classifying the amazon customers reviews text as positive and negative

data["sentiment"] = data["reviews.rating"]>=4
data["sentiment"] = data["sentiment"].replace([True , False] , ["Positive" , "Negative"])

#see frequency distribution of tags

get_ipython().run_line_magic('matplotlib', 'inline')
carrier_count = data["sentiment"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of TAG')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()

data["sentiment"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()

#preprocess data set to remove punctuations and repeating characters/words

import string
import re
english_punctuations = string.punctuation
punctuations_list = english_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def processPost(text):
    text = re.sub('@[^\s]+', ' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text= remove_punctuations(text)
    text=remove_repeating_char(text)

    return text

data["reviews.text"] = data["reviews.text"].apply(lambda x: processPost(x))

#tokenization and removal of stopwords

from nltk.tokenize import RegexpTokenizer
from collections import Counter 
tokenizer = RegexpTokenizer(r'\w+')
data["reviews.text"] = data["reviews.text"].apply(tokenizer.tokenize)

data["reviews.text"].head()

from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')

stopwords_list

data["reviews.text"]=data["reviews.text"].apply(lambda x: [item for item in x if item not in stopwords_list])

data["reviews.text"].head()

all_words = [word for tokens in data["reviews.text"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data["reviews.text"]]

VOCAB = sorted(list(set(all_words)))

print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

#see top words in data set

counter = Counter(all_words)

counter.most_common(25)

counted_words = Counter(all_words)

words = []
counts = []
for letter, count in counted_words.most_common(25):
    words.append(letter)
    counts.append(count)

import matplotlib.cm as cm
from matplotlib import rcParams
colors = cm.rainbow(np.linspace(0, 1, 10))
rcParams['figure.figsize'] = 20, 10

plt.title('Top words in reviews.text')
plt.xlabel('Count')
plt.ylabel('Words')
plt.barh(words, counts, color=colors)

#feature extraction by using TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features =5000)

#prepare features for training and testing

unigramdataGet= word_vectorizer.fit_transform(data['reviews.text'].astype('str'))
unigramdataGet = unigramdataGet.toarray()
vocab = word_vectorizer.get_feature_names()
text_revievs_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
text_revievs_features[text_revievs_features>0] = 1

text_revievs_features.head(20)

data.reset_index(drop=True, inplace=True)
data=data.drop(columns=['reviews.text','reviews.rating'])

#combine all extracted features

Final_data = pd.concat([text_revievs_features, data['sentiment']], axis=1)
Final_data.head()

#encoding positive tag as 1

Final_data['sentiment'] = Final_data['sentiment'].replace({'Postive': 1})

#encoding Negative tag as 0

Final_data['sentiment'] = Final_data['sentiment'].replace({'Negative': 0})

Final_data.head(20)

y=Final_data.sentiment
X=Final_data.drop(columns=['sentiment'])

#split data set into 70% training and 30% testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

#Decision Trees Algorithm

from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier(random_state=10, max_depth=13)
DTC= DTC.fit(X_train , y_train)
DTC


#accuracy of Decision Trees Algorithm

y_pred1 = DTC.predict(X_test)
print('Accuracy score= {:.2f}'.format(DTC.score(X_test, y_test)))

#ROC curve of Decision Trees Algorithm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

#Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=200,max_depth=35, random_state=200,max_leaf_nodes=200)
Ran_For= Ran_For.fit(X_train , y_train)
Ran_For

#accuracy of Random Forest Algorithm

y_pred1 = Ran_For.predict(X_test)
print('Accuracy score= {:.2f}'.format(Ran_For.score(X_test, y_test)))

#ROC curve of Random Forest Algorithm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

#RidgeClassifier Algorithm

from sklearn.linear_model import RidgeClassifier
RC= RidgeClassifier()
RC= RC.fit(X_train , y_train)
RC

#accuracy of RidgeClassifier Algorithm

y_pred1 = RC.predict(X_test)
print('Accuracy score= {:.2f}'.format(RC.score(X_test, y_test)))

#ROC curve of RidgeClassifier Algorithm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

#Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR= LR.fit(X_train , y_train)
LR

#Accuracy of Logistic Regression Algorithm

y_pred1 = LR.predict(X_test)
print('Accuracy score= {:.2f}'.format(LR.score(X_test, y_test)))

#ROC curve of Logistic Regression Algorithm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

#Passive Aggressive Classifier Algorithm

from sklearn.linear_model import PassiveAggressiveClassifier
PC= PassiveAggressiveClassifier()
PC= PC.fit(X_train , y_train)
PC

#accuracy of Passive Aggressive Classifier Algorithm

y_pred1 = PC.predict(X_test)
print('Accuracy score= {:.2f}'.format(PC.score(X_test, y_test)))

#ROC curve of Passive Aggressive Classifier Algorithm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

#comparison of all algorithms

from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Deatiled Performance of the all models")
x.field_names = ["Model", "Accuracy"]

x.add_row(["Decision Trees Algorithm", 0.93])
x.add_row(["Random Forest Algorithm", 0.93])
x.add_row(["RidgeClassifier Algorithm", 0.93])
x.add_row(["LogisticRegression Algorithm", 0.94])
x.add_row(["PassiveAggressiveClassifier Algorithm", 0.92])
print(x)
print('\n')

#most accuracy and fast run time

x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["LogisticRegression Algorithm",0.94])
print(x)
print('\n')

#training all the data on logistic regression for predictions

from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR= LR.fit(X , y)
LR

#predictions on output file

review = input("Please enter Amazom Customer review text:")

new_data=pd.DataFrame(
    {
        "review":[review]
    }
)

new_data.head()

#selection of the interested columns for feature extraction

for_predictions = pd.DataFrame() 
for_predictions['review']=new_data['review']

#applying processpost function for preprocessing

for_predictions['review'] = for_predictions['review'].apply(lambda x: processPost(x))

for_predictions['review']

#tokenization and stop words removal

for_predictions['review'] = for_predictions['review'].apply(tokenizer.tokenize)

for_predictions['review']

for_predictions['review']=for_predictions['review'].apply(lambda x: [item for item in x if item not in stopwords_list])

for_predictions['review']

#transforming the prediction data into vector with same word_vectorizer

x = word_vectorizer.transform(for_predictions['review'].astype('str'))

#getting predictions with trained model of LR

LR.predict(x)
probabilities =LR.predict_proba(x)

#probabilities of positive

class_one=probabilities[:,1]
class_one_D=pd.DataFrame(class_one, columns=['Positive'])
class_one_D['Positive'].head()

class_one_D['Positive'] = class_one_D['Positive'].round(2)
class_one_D['Positive'].head()

class_one_D['Positive'] = (class_one_D['Positive'].sub(class_one_D['Positive'].astype(int))).mul(100).astype(int)
class_one_D['Positive'].head()

#probabilities of negative

class_two=probabilities[:,0]
class_two_D=pd.DataFrame(class_two, columns=['Negative'])
class_two_D['Negative'].head()

class_two_D['Negative'] = class_two_D['Negative'].round(2)
class_two_D['Negative'].head()

class_two_D['Negative'] = (class_two_D['Negative'].sub(class_two_D['Negative'].astype(int))).mul(100).astype(int)

class_two_D['Negative'].head()

pred=LR.predict(x)
prediction=pd.DataFrame(pred, columns=['Predictions'])

#showing predictions with text

new_data['Predictions']=prediction['Predictions']
new_data['Predictions']= new_data['Predictions'].replace({0:'Negative'})
new_data['Predictions']= new_data['Predictions'].replace({1:'Positive'})
new_data['probability_Postive_review']=class_one_D['Positive']
new_data['probability_Negative_review']=class_two_D['Negative']

new_data_=new_data[['review', 'Predictions','probability_Postive_review','probability_Negative_review']]

new_data_

#saving  the predictions in CSV file

new_data.to_csv('prediction.csv')
