# 31-march-2022

# Make a script that gets phrases from json and sends back the 5 most simular

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from nltk.tokenize import word_tokenize
import json

# Opening JSON file
f = open('data.json')
data = json.load(f)
sentences = data

# class object similar
class similar:

    def __init__(self, sente, score):
        self.sente = sente
        self.score = score

    def __repr__(self):
        return '{' + self.sente + ', ' + str(self.score) + '}'



# Tokenization of each document
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))
# print(tokenized_sent)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# use tensorflow module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
# print ("module %s loaded" % module_url)

sentence_embeddings = model(sentences)

# write input in here
query = "Are you a man?"
print("question = ", query)
query_vec = model([query])[0]

sentenceList = []
# print only the 5 best sentences
for sent in sentences:
  sim = cosine(query_vec, model([sent])[0])
  sentenceList.append(similar(sent, sim))
  # print("Sentence = ", sent, "; similarity = ", sim)
sentenceList.sort(key=lambda x: x.score, reverse=True)
# print(sentenceList)

for i in range(5):
    print("Sentence = ", sentenceList[i].sente, "; similarity = ", sentenceList[i].score)

def changeString(input):
    lower = input.lower()
    replace = lower.replace("you", "I")
    return replace

def form(firstSentence):
    string1 = " I do not know how " + changeString(query)
    string2 = " but I do know how " + changeString(firstSentence)
    output = string1 + string2
    return output

if sentenceList[0].score > 0.6:
    firstSentence = sentenceList[0].sente
    print(form(firstSentence))
    print("I do not know what you mean..")
else:
    print("I do not know what you mean..")
