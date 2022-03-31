import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

sentences = ["I ate dinner.",
       "We had a three-course meal.",
       "Brad came to dinner with us.",
       "He loves fish tacos.",
       "In the end, we all felt like we ate too much.",
       "We all agreed; it was a magnificent evening."]

# Tokenization of each document
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))
#print(tokenized_sent)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# import
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
print(tagged_data)

## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)

'''
vector_size = Dimensionality of the feature vectors.
window = The maximum distance between the current and predicted word within a sentence.
min_count = Ignores all words with total frequency lower than this.
alpha = The initial learning rate.
'''

## Print model vocabulary
print(model.wv.key_to_index)

test_doc = word_tokenize("I had pizza and pasta".lower())
test_doc_vector = model.infer_vector(test_doc)
model.docvecs.most_similar(positive = [test_doc_vector])

'''
positive = List of sentences that contribute positively.
'''

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model(sentences)
print(sentence_embeddings)

model.en
#query = "I had pizza and pasta"
#query_vec = model.encode([query])[0]

#for sent in sentences:
#  sim = cosine(query_vec, model.encode([sent])[0])
#  print("Sentence = ", sent, "; similarity = ", sim)