#!/usr/local/bin/python
#Making Word Vectors of 7 books of Harry Potter, Analysing them and searching for semantic 'meaning' similarities.

#IMPORTING

# is a pseudo-module which can be use to enable new language features which are not compatible with the current interpreter.
# Connecting python 2.75 and 3 with eachother. 
from __future__ import absolute_import, division, print_function

#word/text encoding for..
import codecs
 
#is being used at the book_filenames part, to 
import glob 

#concurrency
import multiprocessing

#dealing with the operation system, like reading a file, going to/making the path 'quickly' 
import os

#Pretty Printing, lists etc. 
import pprint

#REgular Expressions, pulling parts out of texts, making all lowercase etc, differnt functions to strip stuff out of content. \S anything but a Space, \w any character etc. 
import re

#Natural Language Tool Kit Modul. Processing Natural Language. Process of your computer to understand natural language. Processing it too numbers what the computer can understand.
import nltk

#Word2vec/gensim can automatically extract semantic topics from documents, designed to process raw, unstructured digital/plain texts. (corpus, vector, model) > writing text away as vector/representation of the words as numbers. 
import gensim.models.word2vec as w2v

#dimensionality reduction, 3+ dimensions, plot them on a 2d graph, that happens here.  
import sklearn.manifold

#math library
import numpy as np

#plotting library
import matplotlib.pyplot as plt

#parse pandas as pd/..
import pandas as pd

#importing seaborn for visualization
import seaborn as sns






#TRAINING/MAKING WORD VECTORS

#cleans data with NLTK 
nltk.download("punkt") #pre-trained (sentence) tokenizer detect sentence boundaries, tokenizing, split a string into a list of sentences/words/characters. 
nltk.download("stopwords") #pre-trained, stopwords like and, a, of, the, is, are, as, will be filtered from the text/corpus. (this is a universal list of stopwords) no semantic meaning so, delete them with nltk. 

book_filenames = sorted(glob.glob("/.../Bookfiles/*.txt"))

print("Found books:")
book_filenames

#Starts with u, because it is unicode string which we want to convert into a format which I can read easily
corpus_raw = u""
for book_filename in book_filenames: 
#This shows that it is reading it, {0} is a string that shows the bookfile name, 
    print("Reading '{0}'...".format(book_filename))
    #codecs library to open the filename and convert it into utf8    
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    #this shows again that {0} becomes the amount of corpus_raw (How many/what length the corpus raw has) as characters. 
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()
    
#Turning words into tokens by loading punkt into memory, split the corpus into sentences using the NLTK punkt english.pickle which is a bit string that has a pre-learned punkt in it. 
#this is gonna be loaded in the tokenizer variable.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#using variable tokenizer to tokenize the corpus_raw set before. 
raw_sentences = tokenizer.tokenize(corpus_raw)
    
#raw_sentence variable will be converted into a wordlist. 
def sentence_to_wordlist(raw):
	#removes unnessecary characters, delete hyphens.
    clean = re.sub("[^a-zA-Z]"," ", raw)
    #split into words
    words = clean.split()
    #returns a list of words.
    return words

#empty sentence list
sentences = []
#sentence where each word is tokenized.
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

#prints the amount of tokens Each sentence as 1 token. 
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))




#BUILDS THE WORD2VEC MODEL
#Dimentionality of the word-vectors, the more dimensions, the more computationally expensive to train. 
#also more accurate, more dimensions = more generalized. number of features
num_features =2000

#The smallest set of words we want to recognize when we convert to vector. 
min_word_count = 3

#number of threads running paralell, at the same time, multiprocessing library.
#more workers, the faster we train. 
num_workers = multiprocessing.cpu_count()

#size of how many it looks at at a time. for example. looking at blocks of 7 words at a time. 
context_size = 7

#downsample setting for frequent words. 
#when it notices a lot of frequent words, it shouldn't look at it constently
#this should be any number between 0 - 1e5. 
#this actually questions: how often do you want to look at the same word? 
downsampling = 1e-3

#random number generator. 
#looks which part we take in the text to look at to convert it into vectors. 
#good for de-bugging + deterministics. 
seed = 2


epochs = 20


potter2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

#builds vocabulary using variable sentences, loading the corpus into memory, we build our model but didnt trained it yet 
potter2vec.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))




#TRAINS THE MODEL

potter2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=epochs)

#making path/directory called trained using the os module. 
if not os.path.exists("trained"):
    os.makedirs("trained")

#string which also prints num_features+epochs as time
filename = os.path.join("trained", "potter2vec{}_{}.w2v".format(str(num_features),str(epochs)))

potter2vec.save(filename)











#LOADING MODEL
potter2vec = w2v.Word2Vec.load(filename)







