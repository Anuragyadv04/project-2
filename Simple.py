
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('data.csv', encoding='latin1')

reviews = data['Sentence'].tolist()
labels = data['Sentiment'].tolist()


X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()


X_train_bow = vectorizer.fit_transform(X_train)


X_test_bow = vectorizer.transform(X_test)


classifier = LogisticRegression(max_iter=1000)  
classifier.fit(X_train_bow, y_train)


y_pred = classifier.predict(X_test_bow)

accuracy = classifier.score(X_test_bow, y_test)
print("Accuracy:", accuracy)



new_data = pd.read_csv('Datb.csv', encoding='latin1')
new_sentences = new_data['Sentence'].tolist()
new_sentiments = new_data['Sentiment'].tolist()


X_test_bow = vectorizer.transform(new_sentences)


y_pred = classifier.predict(X_test_bow)

accuracy = classifier.score(X_test_bow, new_sentiments)
print("Accuracy on new data set:", accuracy)

