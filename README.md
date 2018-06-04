## Tracking, Testing and Measuring Accuracy in a Trained Word2Vec Model

This project shows and explores a preliminary study for training and measuring a Word2Vec model that questions: How does one make and measure if one has a good word2vec model for vector representation? 
This is answered in a effectively, compressed print which shows 16 scatter plots as a spatial map to track and test the outcome of eight trained w2v sets on its accuracy. 


This project hosts 4 files for Tracking, Testing and Measuring Accuracy in a Trained Word2Vec Model. 


# Word2Vec_Potter2Vec_OnlyTrainModel.py
> Code to train a W2V model

# Word2Vec_Potter2Vec_OnlyLoadModel.py
> Code to only Load a W2V model in the Terminal

# ScatterPlotBig.ipynb
> Makes a Big Scatter Plot of a trained model

# ScatterPlotRegion.ipynb
> Makes a Region Scatter Plot of a trained model 

# Folder Bookfiles
> Bookfiles folder consists of all the data the w2v models are trained with: 

Data fed as .TXT files: 
Novel Series of Harry Potter 
1. Harry Potter and the Philosopher's Stone words: 81448, 
2. Harry Potter and the Chamber of Secrets words: 89942, 
3. Harry Potter and the Prisoner of Azkaban words: 113356,  
4. Harry Potter and the Goblet of Fire words: 130318, 
5. Harry Potter and the Order of the Phoenix words: 116834,  
6. Harry Potter and the Half-Blood Prince words: 146511, 
7. Harry Potter and the Deathly Hallows words: 135464.

total amount of words: 817.636

# Folder trained
> Consists of PotterW2V trained models: 
potter2vec100_1.w2v
potter2vec100_10.w2v

potter2vec300_1.w2v
potter2vec300_10.w2v

potter2vec500_1.w2v
potter2vec500_2.w2v
potter2vec500_5.w2v
potter2vec500_10.w2v

potter2vec1000_1.w2v
potter2vec1000_1.w2v.trainables.syn1neg.npy
potter2vec1000_1.w2v.wv.vectors.npy
potter2vec1000_10.w2v
potter2vec1000_10.w2v.trainables.syn1neg.npy
potter2vec1000_10.w2v.wv.vectors.npy

potter2vec2000_20.w2v
potter2vec2000_20.w2v.trainables.syn1neg.npy
potter2vec2000_20.w2v.wv.vectors.npy

potter2vec3000_30.w2v
potter2vec3000_30.w2v.trainables.syn1neg.npy
potter2vec3000_30.w2v.wv.vectors.npy



### Getting started

1.  Install all libraries: 
> codecs
> import glob 
> multiprocessing
> os
> pprint
> re
> nltk
> gensim.models.word2vec as w2v
> sklearn.manifold
> numpy 
> matplotlib.pyplot 
> pandas
> seaborn as sns



### TRAIN YOUR OWN DATASET: 
 
1.  Copy or clone the file `Word2Vec_Potter2Vec_OnlyTrainModel.py` for training your own dataset

* fill your bookfiles folder with the .txt files you want your model to train on.
* Identify where you've located your bookfiles folder.
* Change Num_features and Epoch into desired amount. 
* change Potter2vec into desired name 

> Run script
> Wait until trained and saved (can take a while depending on your set amount of num_features and Epoch) 
