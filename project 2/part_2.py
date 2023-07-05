import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data from CSV file
data = pd.read_csv('data.csv')

# Extract reviews and labels from the DataFrame
reviews = data['Sentence'].tolist()
labels = data['Sentiment'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Create an instance of the CountVectorizer class
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform the training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the testing data using the fitted vectorizer
X_test_bow = vectorizer.transform(X_test)

# Train a classifier using Random Forest
classifier = RandomForestClassifier()
classifier.fit(X_train_bow, y_train)

# Predict sentiment labels for the testing data
y_pred = classifier.predict(X_test_bow)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
