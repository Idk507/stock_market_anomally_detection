import nltk 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from sklearn.metrics import mean_squared_error
from tqdm import tqdm 
import statsmodels.api as sm
from math import sqrt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")
import unicodedata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.metrics import accuracy_score


import nltk
nltk.download('vader_lexicon')

df = pd.read_excel('./tweet data/stock_tweets_selected_automobile.xlsx')


# Plot distribution of tweets over time
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Tweet'].resample('D').count().plot(title='Daily Tweet Count')
plt.show()

df['Stock Name'].value_counts().plot(kind='bar')

# Explore the distribution of tweet lengths
df['Tweet Length'] = df['Tweet'].apply(len)
plt.hist(df['Tweet Length'], bins=30, edgecolor='black')
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

# Explore the most common words in tweets
from wordcloud import WordCloud

# Combine all tweets into a single string
all_tweets = ' '.join(df['Tweet'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_tweets)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud of Tweets')
plt.show()



# Explore the distribution of tweets across different companies
company_counts = df['Company Name'].value_counts()
company_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Tweets Across Companies')
plt.xlabel('Company Name')
plt.ylabel('Number of Tweets')
plt.show()

# Display the counts of tweets for each company
print("Counts of Tweets for Each Company:")
print(company_counts)

# Explore the distribution of tweets across different stock names
stock_counts = df['Stock Name'].value_counts()
stock_counts.plot(kind='bar', color='lightcoral')
plt.title('Distribution of Tweets Across Stock Names')
plt.xlabel('Stock Name')
plt.ylabel('Number of Tweets')
plt.show()

# Display the counts of tweets for each stock name
print("\nCounts of Tweets for Each Stock Name:")
print(stock_counts)

null_values = df.isnull().sum()
print("Null Values in the Dataset:")
print(null_values)

# Check for duplicates in the dataset
duplicate_rows = df[df.duplicated()]
print("\nNumber of Duplicate Rows:", duplicate_rows.shape[0])

# Perform sentiment analysis using a library like TextBlob or VaderSentiment
from textblob import TextBlob

# Function to classify sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis to the 'Tweet' column
df['Sentiment'] = df['Tweet'].apply(get_sentiment)



# Visualize the distribution of sentiments
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Distribution of Sentiments in Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()


analyzer = SentimentIntensityAnalyzer()
analyzer

df['Sentiment'].value_counts()

df['Sentiment_score'] = df['Tweet'].apply(lambda text: analyzer.polarity_scores(text)['compound'])


mean_sentiment = df['Sentiment_score'].mean()
std_sentiment = df['Sentiment_score'].std()
df['Z-Score'] = (df['Sentiment_score'] - mean_sentiment) / std_sentiment


threshold = 2.0

anomalies = df[df['Z-Score'].abs() > threshold]

print("Sentiment Analysis Results:")
print(df)

print("\nAnomalies:")
print(anomalies)


# Word Cloud for Positive, Neutral, and Negative Sentiments
def plot_wordcloud(sentiment):
    words = ' '.join(df[df['Sentiment'] == sentiment]['Tweet'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

plot_wordcloud('Positive')




X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(tfidf_train, y_train)


y_pred = naive_bayes.predict(tfidf_test)


print("Classification Report:\n", classification_report(y_test, y_pred))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
# Function to plot ROC-AUC curve
def plot_roc_curve(y_true, y_prob, classes):
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green'])
    plt.figure(figsize=(8, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.show()
    

# Plotting the Confusion Matrix
plot_confusion_matrix(y_test, y_pred, classes=['Negative', 'Neutral', 'Positive'])

# Plotting the Classification Report
plt.figure(figsize=(8, 4))
sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T, annot=True, cmap="Blues")
plt.title('Classification Report')
plt.show()

# Plotting the ROC-AUC Curve
y_prob = naive_bayes.predict_proba(tfidf_test)
plot_roc_curve(y_test, y_prob, classes=['Negative', 'Neutral', 'Positive'])


sent_df = df.copy()
sent_df["Sentiment_score"] = ''
sent_df["Negative"] = ''
sent_df["Neutral"] = ''
sent_df["Positive"] = ''
sent_df.head()

tweet = df['Tweet']

sent_df.T




sentiment_analyzer = SentimentIntensityAnalyzer()

for indx, row in sent_df.iterrows():
    try:
        sentence_i = unicodedata.normalize('NFKD', row['Tweet'])
        sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
        sent_df.at[indx, 'Sentiment_score'] = sentence_sentiment['compound']
        sent_df.at[indx, 'Negative'] = sentence_sentiment['neg']
        sent_df.at[indx, 'Neutral'] = sentence_sentiment['neu']
        sent_df.at[indx, 'Positive'] = sentence_sentiment['pos']
    except TypeError:
        print(sent_df.loc[indexx, 'Tweet'])
        print(indx)
        break


sent_df.head()

mean_sentiment = sent_df['Sentiment_score'].mean()
std_sentiment = sent_df['Sentiment_score'].std()
sent_df['Z-Score'] = (sent_df['Sentiment_score'] - mean_sentiment) / std_sentiment

threshold = 2.0

anomalies = sent_df[sent_df['Z-Score'].abs() > threshold]

print("Sentiment Analysis Results:")
print(sent_df)

print("\nAnomalies:")
print(anomalies)

sent_df.reset_index(inplace=True)


sent_df['Date'] = pd.to_datetime(sent_df['Date'])
print(sent_df.head())
sent_df['Date'] = sent_df['Date'].dt.date
print(sent_df['Date'].head())



X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(tfidf_train, y_train)


y_pred = naive_bayes.predict(tfidf_test)


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    
    
# Function to plot ROC-AUC curve
def plot_roc_curve(y_true, y_prob, classes):
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green'])
    plt.figure(figsize=(8, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
stocks = pd.read_csv('./tweet data/automobile_stocks.csv')
stocks.head()
'''Date	Open	High	Low	Close	Adj Close	Volume	Stock Name
0	2022-10-26	219.399994	230.600006	218.199997	224.639999	224.639999	85012500	TSLA
1	2022-10-27	229.770004	233.809998	222.850006	225.089996	225.089996	61638800	TSLA
2	2022-10-28	225.399994	228.860001	216.350006	228.520004	228.520004	69152400	TSLA
3	2022-10-31	226.190002	229.850006	221.940002	227.539993	227.539993	61554300	TSLA
4	2022-11-01	234.050003	237.399994	227.279999	227.820007	227.820007	62688800	TSLA'''

stocks.shape
#(4723, 8)

stock_names_to_select = ['TSLA', 'NIO', 'XPEV', 'F']


selected_rows = stocks[stocks['Stock Name'].isin(stock_names_to_select)]
stock_df.dtypes
'''Date           object
Open          float64
High          float64
Low           float64
Close         float64
Adj Close     float64
Volume          int64
Stock Name     object
dtype: object'''

final_df = stock_df.merge(twitt_df, on='Date', how='left')
final_df.head()
'''Date	Open	High	Low	Close	Adj Close	Volume	Sentiment_score
0	2022-10-26	219.399994	230.600006	218.199997	224.639999	224.639999	85012500	NaN
1	2022-10-27	229.770004	233.809998	222.850006	225.089996	225.089996	61638800	NaN
2	2022-10-28	225.399994	228.860001	216.350006	228.520004	228.520004	69152400	NaN
3	2022-10-31	226.190002	229.850006	221.940002	227.539993	227.539993	61554300	NaN
4	2022-11-01	234.050003	237.399994	227.279999	227.820007	227.820007	62688800	NaN'''


df.reset_index(inplace=True)


df['Date'] = pd.to_datetime(df['Date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

#adding techinacal indicators 

def get_tech_ind(data):
    data['MA7'] = data.iloc[:,4].rolling(window=7).mean() 
    data['MA20'] = data.iloc[:,4].rolling(window=20).mean() 

    data['MACD'] = data.iloc[:,4].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()
    #This is the difference of Closing price and Opening Price

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:,4].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:,4] - 1)

    return data


tech_df = get_tech_ind(final_df)
dataset = tech_df.iloc[20:,:].reset_index(drop=True)
dataset.head()


tech_df

'''
Date	Open	High	Low	Close	Adj Close	Volume	Sentiment_score	MA7	MA20	MACD	20SD	upper_band	lower_band	EMA	logmomentum
0	2022-10-26	219.399994	230.600006	218.199997	224.639999	224.639999	85012500	NaN	NaN	NaN	5.240005	NaN	NaN	NaN	224.639999	5.410038
1	2022-10-27	229.770004	233.809998	222.850006	225.089996	225.089996	61638800	NaN	NaN	NaN	3.878271	NaN	NaN	NaN	224.977497	5.412048
2	2022-10-28	225.399994	228.860001	216.350006	228.520004	228.520004	69152400	NaN	NaN	NaN	4.510737	NaN	NaN	NaN	227.430002	5.427238
3	2022-10-31	226.190002	229.850006	221.940002	227.539993	227.539993	61554300	NaN	NaN	NaN	4.194962	NaN	NaN	NaN	227.504246	5.422921
4	2022-11-01	234.050003	237.399994	227.279999	227.820007	227.820007	62688800	NaN	NaN	NaN	2.689250	NaN	NaN	NaN	227.715623	5.424157'''

dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill()])

datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)

merged_df = pd.merge(stocks, df, on=['Date', 'Stock Name'], how='inner')
merged_df.head()

'''	Date	Open	High	Low	Close	Adj Close	Volume	Stock Name	Tweet	Company Name	Tweet Length	Sentiment	Sentiment_score	Z-Score
0	2021-09-30	260.333344	263.043335	258.333344	258.493347	258.493347	53868000	TSLA	#LottoFriday Watchlist: short &amp; sweet\n\n$...	Tesla, Inc.	240	Positive	0.8478	1.487776
1	2021-09-30	260.333344	263.043335	258.333344	258.493347	258.493347	53868000	TSLA	CORRECTION UPDATE\n\nUPDATE on Q3 Delivery Est...	Tesla, Inc.	296	Neutral	-0.1531	-0.747275
2	2021-09-30	260.333344	263.043335	258.333344	258.493347	258.493347	53868000	TSLA	FREE #OPTIONS Ideas 🤯\n\nScale out when above ...	Tesla, Inc.	317	Positive	0.9083	1.622875
3	2021-09-30	260.333344	263.043335	258.333344	258.493347	258.493347	53868000	TSLA	California DMV today issued autonomous vehicle...	Tesla, Inc.	272	Positive	0.0000	-0.405396
4	2021-09-30	260.333344	263.043335	258.333344	258.493347	258.493347	53868000	TSLA	@chamath Appreciate the clarification @chamath...	Tesla, Inc.	196	Positive	0.4019	0.49206'''

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


stock = pd.read_csv('./tweet data/merged_df.csv')
merged_df['Stock Name'].value_counts()
ano_df = pd.DataFrame(merged_df[['Date','Close','Stock Name']])
ano_df.set_index('Stock Name',inplace=True)
ano_df.head()
'''	Date	Close
Stock Name		
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347'''

ano_df['Stock Name'].value_counts()
data_training = pd.DataFrame(ano_df['Close'][0:int(len(ano_df)*0.70)])
data_testing = pd.DataFrame(ano_df['Close'][int(len(ano_df)*0.70):int(len(ano_df))])

'''(        Close
 0  258.493347
 1  258.493347
 2  258.493347
 3  258.493347
 4  258.493347,
             Close
 33810  325.733337
 33811  325.733337
 33812  325.733337
 33813  325.733337
 33814  325.733337)'''
 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_training_array
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
x_train,y_train = np.array(x_train), np.array(y_train)

x_train.shape,y_train.shape

model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 60,activation = 'relu',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units = 80,activation = 'relu',return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units = 120,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()
model.fit(x_train,y_train,epochs=10,batch_size=32)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df.shape

input_data = scaler.fit_transform(final_df)
input_data.shape

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
    
x_test,y_test = np.array(x_test), np.array(y_test)
x_test.shape,y_test.shape

y_predict = model.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_test, y_predict)
print(f'Mean Absolute Error: {mae:.4f}')
mse = mean_squared_error(y_test, y_predict)
print(f'Mean Squared Error: {mse:.4f}')


#autoencoder model

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(30, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')
def normalize_window_roll(data, window_size):
    X = []
    Y = []
    for i in range(0,len(data) - window_size,5):
        x_window = data.iloc[i:i+window_size]
        
        # Normalize the x_window and y_window
        scaler = MinMaxScaler()
        x_window = scaler.fit_transform(np.array(x_window).reshape(-1, 1))
        
        X.append(x_window)
    X =np.squeeze(np.array(X), axis=2)

    return X

X= normalize_window_roll(ano_df['Close'],30)

train_size = int(0.7*X.shape[0])
test_size = int(0.1*X.shape[0])
val_size = int(0.2*X.shape[0])

X_train = X[:train_size]
X_test = X[train_size:train_size+test_size]
X_val = X[train_size+test_size:train_size+test_size+val_size]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2,
                                                  mode='min')
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(X_train,X_train, epochs=40,
                    validation_data=(X_val, X_val),
                    batch_size=16,
                    callbacks=[early_stopping])
reconstructions = autoencoder.predict(X_train)
train_loss = tf.keras.losses.mae(reconstructions, X_train)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
reconstructions = autoencoder.predict(X_test)
test_loss = tf.keras.losses.mae(reconstructions, X_test)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

preds = predict(autoencoder, X_test, threshold)

anomalous_test_data = X_test[np.where(preds==False)]

encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()
 
for p in [1,10]:
  plt.plot(anomalous_test_data[p], 'b')
  plt.plot(np.arange(0,X.shape[1]),decoded_data[p], 'r')
  plt.fill_between(np.arange(X.shape[1]), decoded_data[p], anomalous_test_data[p], color='lightcoral')
  plt.legend(labels=["Input", "Reconstruction", "Error"])
  plt.show()
  

from sklearn.ensemble import IsolationForest

ano_df.head()

'''	Date	Close
Stock Name		
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347
TSLA	2021-09-30	258.493347'''

data = ano_df.copy()

data.reset_index(inplace=True)
ano_df.columns

ano_df.reset_index(inplace=True)
unique_stock_names = ano_df['Stock Name'].unique()

#array(['TSLA', 'F', 'NIO', 'XPEV'], dtype=object
model = IsolationForest(contamination=0.05, random_state=42)
anomalies_info = {}
data
for stock_name in unique_stock_names:
    stock_data = data[data['Stock Name'] == stock_name]
    
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    model.fit(close_prices)

    anomaly_predictions = model.predict(close_prices)

    stock_data['Anomaly'] = anomaly_predictions

    anomalies_info[stock_name] = {
        'Date': stock_data['Date'][anomaly_predictions == -1],
        'Close Price': stock_data['Close'][anomaly_predictions == -1]
    }

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
    plt.scatter(stock_data['Date'][anomaly_predictions == -1], stock_data['Close'][anomaly_predictions == -1], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Anomaly Detection for {stock_name} Stock')
    plt.legend()
    plt.show()


for stock_name, info in anomalies_info.items():
    print(f'Anomalies for {stock_name} stock:')
    anomalies_df = pd.DataFrame(info)
    print(anomalies_df)
    print("\n")
    