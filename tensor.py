import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from nltk.tokenize import word_tokenize

sentences = ["Can you make coffee?.",
       "Do you like coffee?",
       "What do you eat?",
       "What is your favorite music?",
       "Are you happy?",
       "Why did the chicken cross the road?"]

# Tokenization of each document
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))
#print(tokenized_sent)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

sentence_embeddings = model(sentences)
query = "What do you drink?"
print("question = ", query)
query_vec = model([query])[0]

for sent in sentences:
  sim = cosine(query_vec, model([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)