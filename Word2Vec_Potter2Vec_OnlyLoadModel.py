#!/usr/local/bin/python
#Making Word Vectors of 5 books of Game Of Thrones, Analysing them and searching for semantic'meaning' similarities.

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

#visualization/..
import seaborn as sns



#OPEN
potter2vec = w2v.Word2Vec.load(os.path.join("trained", "potter2vec100_1.w2v")) #insert the desired trained model you want to load.





