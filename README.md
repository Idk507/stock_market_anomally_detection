This research presents a multifaceted methodology for sentiment analysis in the stock market, coupled with anomaly detection using a variety of machine learning techniques. The study encompasses six major steps, starting with data collection utilizing NLTK, Vader sentiment, NumPy, Pandas, TensorFlow, scikit-learn, and Matplotlib. The sentiment analysis focuses on four prominent automobile companies—TSLA, F, NIO, and XPEV—employing TextBlob for polarity determination.In the second step, a classification model is built using Naive Bayes, evaluating performance metrics like accuracy, recall, F1 score, and precision. A comparative analysis with other models such as Random Forest, Decision Tree, SVM, and KNN is conducted, with SVM emerging as the most accurate. The sentiment scores are normalized, and the sentiment analyzer is integrated into the dataset.The third step involves the integration and visualization of stock data, incorporating technical indicators like closing and opening prices, Bollinger Bands, exponential moving averages, and log momentum. The fourth step explores plotting stock price momentum based on tweet sentiment and analyzing sentiment distribution over time, along with correlation studies between sentiment and stock prices.Moving forward, the research delves into anomaly detection and time series forecasting by implementing a Bi LSTM model for stock prediction. Metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE) reveal the accuracy of the time series forecasting model. An ML model for anomaly detection is created, and anomalies are plotted for all four companies. Future features are generated for the next 30 days, followed by anomaly forecasts for the same duration.This comprehensive approach provides valuable insights into the interplay between sentiment, stock market dynamics, and anomaly detection, offering a robust framework for understanding and predicting market behavior.

## 1. Data Collection:
- `Sources:` Collect historical stock price data for the selected automobile companies (TSLA, F, NIO, XPEV) and relevant financial indicators. Obtain social media data, particularly tweets mentioning the selected companies.
- `Tools and Libraries:` Utilize financial databases, APIs, and web scraping tools for stock data. Leverage social media APIs or web scraping for collecting tweet data.
- `Data Preprocessing:` Clean and preprocess the data, handling missing values, removing duplicates, and ensuring data consistency.
## 2. Sentiment Analysis:
- `Text Processing:` Use Natural Language Processing (NLP) techniques for text processing, including tokenization, stemming, and lemmatization.
- `Sentiment Lexicons:` Employ sentiment lexicons, such as the Vader lexicon, for sentiment analysis on tweet data.
- `Sentiment Classification Models:` Implement sentiment classification models, such as TextBlob, Naive Bayes, or other machine learning algorithms, to categorize tweets into positive, negative, or neutral sentiments.
## 3. Integration with Stock Data:
- `Temporal Alignment:` Align sentiment data with stock price data based on timestamps.
- `Feature Engineering:` Incorporate technical indicators like Bollinger Bands, exponential moving averages, and log momentum into the dataset.
## 4. Anomaly Detection:
- `Machine Learning Models:` Develop an anomaly detection framework using machine learning models such as One-Class SVM, Isolation Forest, or Autoencoders.
- `Threshold Setting:` Determine appropriate threshold values for anomaly detection based on model performance and domain knowledge.
- `Model Evaluation:` Evaluate the performance of the anomaly detection model using metrics like precision, recall, and F1 score.
## 5. Time Series Forecasting:
- `Bi LSTM Model:` Implement a Bi-directional Long Short-Term Memory (Bi LSTM) neural network for time series forecasting.
- `Training and Validation:` Split the dataset into training and validation sets. Train the model on historical data and validate on a separate dataset.
- `Metrics:` Evaluate the time series forecasting model using regression metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE).
## 6. Comparative Analysis:
- `Model Comparison:` Compare the performance of sentiment classification models, anomaly detection models, and time series forecasting models.
- `Statistical Tests:` Use statistical tests to compare accuracies and other relevant metrics across different models.
## 7. Visualization and Interpretation:
- `Visualization Tools:` Utilize tools like Matplotlib and Seaborn for visualizing sentiment trends, stock price movements, and anomalies.
- `Interpretation:` Interpret the results, highlighting key findings and insights derived from the analyses.

## Contributors:  
- @Dhanushkumar [https://github.com/Idk507]
- @Jeevitha [https://github.com/JEEVITHA2512]

