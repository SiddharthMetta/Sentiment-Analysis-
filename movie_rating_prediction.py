import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.backend as kerasBackend
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

import io
import os 

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re


df =pd.read_csv('data.csv')
print(df.shape)


print (df['stars'].value_counts())



# balancing the classes
no_of_samples = 200000


df_stars_1 = resample(df[df.stars==1], replace=True, n_samples=no_of_samples ,random_state=123)
df_stars_2 = resample(df[df.stars==2], replace=True, n_samples=no_of_samples ,random_state=123)
df_stars_3 = resample(df[df.stars==3], replace=True, n_samples=no_of_samples ,random_state=123)
df_stars_4 = resample(df[df.stars==4], replace=True, n_samples=no_of_samples ,random_state=123)
df_stars_5 = resample(df[df.stars==5], replace=True, n_samples=no_of_samples ,random_state=123)

df_sampled = pd.concat([df_stars_1, df_stars_2, df_stars_3, df_stars_4, df_stars_5])
df_sampled.shape

Xdata = df_sampled['text']
Ydata = df_sampled['stars']
(Xdata.shape), (Ydata.shape)



def preprocess(line):
  tokens = word_tokenize(line)

  stop_words = set(stopwords.words("english"))
  words = [w.lower() for w in tokens if not w in stop_words]

  # considering only those with alphabets
  words_split = []
  for i in words:
    words_split += [word for word in re.findall(r"\w+", i)]

  # removing words which are not alphabetic
  words_split = [word for word in words_split if word.isalpha()]

  # removing words which have len < 2
  words_split = [word for word in words_split if len(word) > 2]

  # removing duplicates
  non_duplicate_words = []
  for i in words_split:
    if i not in non_duplicate_words:
      non_duplicate_words.append(i)
      
      
  return non_duplicate_words



data_x = Xdata.apply(lambda x : " ".join(preprocess(x)))

data_x.shape, Ydata.shape

Ydata.value_counts()

df = pd.concat([data_x,Ydata], axis=1)
df = df.dropna()

print (df["stars"].value_counts())


dataX = df["text"]
dataY = df["stars"]
dataX2 = [x for x in dataX if x]
dataY.value_counts()


df3 = df.groupby('stars').head(10000).reset_index(drop=True)


dataX = df3["text"]
dataY = df3["stars"]


Y = to_categorical(dataY)

num_classes = 6

seed = 101
np.random.seed(seed)


(X_train, X_test, Y_train, Y_test) = train_test_split(dataX, Y, test_size=0.3, stratify= dataY, random_state=seed)

max_features = 2000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_words = 2000
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape, X_test.shape)

batch_size = 100
epochs = 20

def getModel(max_features, word_vector_dimension, embedding_matrix):
	np.random.seed(seed)
	kerasBackend.clear_session()

	model = Sequential()

	model.add( Embedding( max_features, word_vector_dimension, input_length = X_train.shape[1], weights = [embedding_matrix] ))
	model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
	model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile( loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
	
	print(model.summary())
	return model

def get_coefficients(word, *arr):
    return word, np.asarray(arr, dtype="float32")

def get_embedding_matrix(embedding_file, max_features=2000):
	embeddings_index = dict(get_coefficients(*i.rstrip().rsplit(' ')) for i in open(embedding_file,encoding="utf-8"))
	print("Number of word vectors: ", len(embeddings_index))

	# getting the embedding matrix
	word_index = tokenizer.word_index
	print (word_index)
	num_words = min(max_features, len(word_index) + 1) # len(word_index)+1 because ???

	all_embeddings = np.stack(embeddings_index.values())
	
	# normalizing data
	embedding_matrix = np.random.normal(all_embeddings.mean(), all_embeddings.std(), (num_words, word_vector_dimension))
	


	l= []
	for word, i in word_index.items():
		if i >= max_features:
			continue
		embedding_vector = embeddings_index.get(word)
		l.append(embedding_vector)
		
		
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	max_features = embedding_matrix.shape[0]

	return max_features, embedding_matrix

embedding_file = "glove.6B.100d.txt"
word_vector_dimension = 100
max_features, embedding_matrix = get_embedding_matrix(embedding_file)


# training the model
model = getModel(max_features, word_vector_dimension, embedding_matrix)
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, batch_size = batch_size, verbose=2)

dataXtest =["this restaurant is amazing", "this is worst restaurant ever", "wow too good restaurant"]
dataXtest = tokenizer.texts_to_sequences(dataXtest)

dataXtest = sequence.pad_sequences(dataXtest, maxlen=max_words)


l = (model.predict_classes(dataXtest, batch_size=batch_size, verbose=0))
output = l.tolist()
print (output)

