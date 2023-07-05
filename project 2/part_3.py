import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

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
X_train_bow = vectorizer.fit_transform(X_train).toarray()

# Transform the testing data using the fitted vectorizer
X_test_bow = vectorizer.transform(X_test).toarray()

# Encode the labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_bow.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_bow, y_train_encoded, epochs=10, batch_size=32)

# Evaluate the model on the testing data
y_test_encoded = label_encoder.transform(y_test)
loss, accuracy = model.evaluate(X_test_bow, y_test_encoded)
print("Accuracy:", accuracy)
