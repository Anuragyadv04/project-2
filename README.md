# project-2 :Financial Sentiment Analysis 

Project:   "Financial Market Sentiment Analysis":


Problem Statement:

The problem is to develop a sentiment analysis system for financial markets that accurately captures and predicts the sentiment of market participants based on various data sources. The goal is to analyze the sentiment of news articles, social media posts, and other relevant textual data to understand the impact of sentiment on stock prices, market trends, and investor behavior. By accurately assessing the sentiment, the system will enable traders, investors, and financial institutions to make informed decisions and gain a competitive advantage in the financial markets. The challenge lies in developing a robust and scalable sentiment analysis model that can handle the vast amount of financial data, adapt to evolving market dynamics, and provide real-time insights for effective decision-making.

 Data Collection
- Gather relevant financial data from various sources such as news articles, social media feeds, financial forums, and company reports. Collect a diverse range of textual data that captures the sentiment of market participants.


 Data Preprocessing
- Clean and preprocess the collected data to remove noise, and irrelevant information, and ensure consistency. Apply techniques like text normalization, tokenization, stop-word removal, and stemming/lemmatization to prepare the textual data for analysis.

 Sentiment Labeling
- Manually or automatically label the preprocessed data with sentiment labels such as positive, negative, or neutral. This can involve using existing sentiment lexicons, employing crowd-sourcing, or training a separate sentiment classifier.

( https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)
(For this project I took the two  labeled  data sets  from Kaggle "data.csv" and "datb.csv" ."datab.csv" is the combination of the two )

 Feature Extraction
- Extract relevant features from the preprocessed textual data. This can include bag-of-words representations, word embeddings (e.g., Word2Vec, GloVe), or more advanced techniques like BERT embeddings. Consider incorporating domain-specific features that capture financial jargon and sentiment indicators.

(For this project I tried out   bag-of-words, Word2Vec, and  GloVe techniques for feature extraction. The bag-of-words approach is being used in Simple.py and The GloVe approach is being used in Advance.py )

 Model Selection and Training
- Choose an appropriate machine learning or deep learning model for sentiment analysis. This can range from traditional models such as Naive Bayes, Support Vector Machines (SVM), or more advanced techniques like Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), or Transformers. Train the selected model using the labeled dataset from Step 3.

( Started with simple as logistic regression, random forest,  and finally advanced methods of neural networks)

 Model Evaluation
- Evaluate the performance of the trained sentiment analysis model using suitable evaluation metrics such as accuracy, precision, recall, and F1-score. Utilize techniques like cross-validation to ensure the model's generalizability and assess its performance on unseen data.

  "Advance.py"- using GloVe to extract features and using Recurrent Neural Networks (RNNs)
  Gave maximum accuracy of 84% on a Data set of datab.csv
  Using data.csv to train gives an accuracy of 69% and on being tested on datb.csv gives an 
  accuracy of 87%
  Using datb.csv to train gives an accuracy of 78% and on being tested on data.csv gives an 
  accuracy of 76%

  "Simple.py" - using bag-of-words to extract features and using Logistic Regression
  Gave maximum accuracy of 80% on a Data set of datab.csv
  Using data.csv to train gives an accuracy of 69% and on being tested on datb.csv gives an 
  accuracy of 84%
  Using datb.csv to train gives an accuracy of 77% and on being tested on data.csv gives an 
  accuracy of 73%

Additionally :
Following things can be done further

Visualization and Interpretation
- Visualize the sentiment analysis results using appropriate charts, graphs, or dashboards. Analyze and interpret the sentiment trends, identify key sentiment drivers, and gain insights into the impact of sentiment on market trends and investor behavior.

Model Refinement and Iteration
- Continuously refine and improve the sentiment analysis model based on feedback and new data. Explore techniques like ensemble methods, hyperparameter tuning, or incorporating additional data sources to enhance the accuracy and relevance of the sentiment analysis.

Step 10: Performance Monitoring and Updates
- Monitor the performance of the sentiment analysis system over time and update the model periodically to adapt to changing market dynamics and evolving sentiment patterns. Stay updated with the latest research and techniques in sentiment analysis to incorporate advancements into the system.







