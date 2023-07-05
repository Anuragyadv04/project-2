import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load data from CSV file
data = pd.read_csv("data.csv")

# Preprocess the data if required (cleaning, tokenization, etc.)

# Split the data into sentences and corresponding sentiment labels
sentences = data["Sentence"].tolist()
sentiments = data["Sentiment"].tolist()

# Encode sentiment labels
label_encoder = LabelEncoder()
encoded_sentiments = label_encoder.fit_transform(sentiments)
num_classes = len(label_encoder.classes_)

# Split the data into training and testing sets
sentences_train, sentences_test, sentiments_train, sentiments_test = train_test_split(
    sentences, encoded_sentiments, test_size=0.2, random_state=42
)

# Train Word2Vec model on the training data
word2vec_model = Word2Vec(sentences_train, vector_size=100, window=5, min_count=1)

# Generate sentence vectors using the trained Word2Vec model
def get_sentence_vector(sentence):
    vector = []
    for word in sentence.split():
        if word in word2vec_model.wv:
            vector.append(word2vec_model.wv[word])
    return np.mean(vector, axis=0) if vector else np.zeros(word2vec_model.vector_size)

train_vectors = [get_sentence_vector(sentence) for sentence in sentences_train]
test_vectors = [get_sentence_vector(sentence) for sentence in sentences_test]

# Convert data to numpy arrays
train_vectors = np.array(train_vectors)
test_vectors = np.array(test_vectors)
sentiments_train = to_categorical(sentiments_train, num_classes)
sentiments_test = to_categorical(sentiments_test, num_classes)

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(word2vec_model.vector_size,)))
model.add(Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_vectors, sentiments_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(test_vectors, sentiments_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)
