import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import SnowballStemmer  # Usamos SnowballStemmer para español
from tensorflow import keras

stemmer = SnowballStemmer("spanish")  # Seleccionamos el idioma español

# Cargar el archivo de intenciones en español
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

nltk.download('punkt')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [stemmer.stem(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words_es.pkl', 'wb'))
pickle.dump(classes, open('classes_es.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [stemmer.stem(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Reformatear la lista de entrenamiento antes de convertirla en un array numpy
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

train_process = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model_es.h5")
