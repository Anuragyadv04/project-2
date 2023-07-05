import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Preprocess the data
texts = data['Sentence'].tolist()
labels = data['Sentiment'].tolist()

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to sequences of tokens
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

# Pad sequences to have the same length
max_sequence_length = 100  # Maximum length of a sentence
sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Encode the labels to numerical values
label_encoder = LabelEncoder()
label_encoder.fit(labels_train)
labels_train = label_encoder.transform(labels_train)
labels_test = label_encoder.transform(labels_test)

# Convert the labels to one-hot encoding
num_classes = len(set(labels))
labels_train = to_categorical(labels_train, num_classes)
labels_test = to_categorical(labels_test, num_classes)

# Load the GloVe word embeddings
embedding_dim = 100  # Dimensionality of the word embeddings
glove_path = 'glove.6B.100d.txt'

embedding_matrix = np.zeros((vocab_size, embedding_dim))
with open(glove_path, encoding='utf-8') as glove_file:
    for line in glove_file:
        word, *vector = line.split()
        if word in tokenizer.word_index:
            embedding_matrix[tokenizer.word_index[word]] = np.array(vector, dtype=np.float32)

# Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(sequences_train, labels_train, validation_data=(sequences_test, labels_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(sequences_test, labels_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
