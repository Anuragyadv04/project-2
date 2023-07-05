import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Train a classifier using Logistic Regression
classifier = LogisticRegression(max_iter=1000)  # Increase the max_iter parameter
classifier.fit(X_train_bow, y_train)

# Predict sentiment labels for the testing data
y_pred = classifier.predict(X_test_bow)
# Evaluate the performance of the classifier
accuracy = classifier.score(X_test_bow, y_test)
print("Accuracy:", accuracy)



# Input new data for prediction
new_reviews = [
    "The Stockmann department store will have a total floor space of over 8,000 square metres and Stockmann 's investment in the project will have a price tag of about EUR 12 million .",
    "Circulation revenue has increased by 5 % in Finland and 4 % in Sweden in 2008 .",
    "It's an average experience."
]

# Transform the new data using the fitted vectorizer
new_reviews_bow = vectorizer.transform(new_reviews)

# Make predictions on the new data
predictions = classifier.predict(new_reviews_bow)

# Print the predictions
for review, prediction in zip(new_reviews, predictions):
    print("Review:", review)
    print("Prediction:", prediction)
    print()