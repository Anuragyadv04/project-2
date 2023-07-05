import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import accuracy_score

# Load the data from the CSV file
data = pd.read_csv('dataa.csv', encoding='latin1')

# Preprocess the data
sentences = data['Sentence'].values
sentiments = data['Sentiment'].values

# Convert sentiment labels to numeric format
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
sentiments = np.array([sentiment_mapping[sentiment] for sentiment in sentiments])

# Split the data into training and testing sets
sentences_train, sentences_test, sentiments_train, sentiments_test = train_test_split(
    sentences, sentiments, test_size=0.2, random_state=42)

# Tokenize the sentences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences_train)
sequences_train = tokenizer.texts_to_sequences(sentences_train)
sequences_test = tokenizer.texts_to_sequences(sentences_test)

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences_train)
sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Convert sentiment labels to one-hot encoded format
num_classes = len(sentiment_mapping)
sentiments_train = to_categorical(sentiments_train, num_classes=num_classes)
sentiments_test = to_categorical(sentiments_test, num_classes=num_classes)

# Create the CNN model
embedding_dim = 100
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(sequences_train, sentiments_train, validation_data=(sequences_test, sentiments_test),
          epochs=10, batch_size=128)

# Evaluate the model
predicted_probabilities = model.predict(sequences_test)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
true_classes = np.argmax(sentiments_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print('Accuracy:', accuracy)

## Load the new dataset
new_data = pd.read_csv('Data.csv', encoding='latin1')
new_sentences = new_data['Sentence'].values
new_sentiments = new_data['Sentiment'].values

# Convert sentiment labels of new dataset to numeric format
new_sentiments = [sentiment_mapping[sentiment] for sentiment in new_sentiments]

# Preprocess the new data
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)

# Predict the sentiments for the new data
predicted_probabilities = model.predict(new_sequences)
predicted_sentiments = np.argmax(predicted_probabilities, axis=1)

# Calculate the accuracy
accuracy = accuracy_score(new_sentiments, predicted_sentiments)
print('Accuracy on new dataset:', accuracy)