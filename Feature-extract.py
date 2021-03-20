'''
Authors - Andrew Vo & Anant Natekar
'''
import nltk
import os
import re
import random  
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

#the below sentence to be run once only. After its downloaded the sentence can be commented
#nltk.download('stopwords')

ps = PorterStemmer() #invoking method for stemming
alldocuments = [] #list for storing the output
wordfreq = {} #defining constants
path = '/Users/nat2147/anant/school/marquette/data mining/mini_newsgroups/misc.forsale' #use this line while testing
#path = '/Users/nat2147/anant/school/marquette/data mining/mini_newsgroups' #use this line while running it fully
for (root, directories, files) in os.walk(path):
    for name in files:
        f = open(root+'/'+name, 'r', errors='ignore')
        content = f.readlines() #reads all the lines of a file
        #clean_content = content.
        for sentence in content:
            tokens = nltk.word_tokenize(sentence) #tokenizes the sentence
            for token in tokens:
                token = token.lower() #lower case
                token = ps.stem(token) #stemming
                token = re.sub(r'\W',' ',token) #removes punctuation
                token = re.sub(r'\s+',' ',token) #removes empty text
                stop_word = set(stopwords.words('english'))
                if token not in stop_word: #removes stop words
                    if token not in wordfreq: #counts number of words
                        wordfreq[token] = 1
                    else:
                        wordfreq[token] += 1
        fileformat = [directories, name, wordfreq]
        f.close()
alldocuments.append(fileformat)

term_dict = {}
term_doc_freq = {} # document-frequency
for x in range(len(alldocuments)):
   for name in alldocuments:
      for token in wordfreq:
          if token not in term_dict:
              term_dict[token] = 1
              feature_id = 0
          if token not in term_doc_freq:
              term_doc_freq[token] = 1
          else:
            term_doc_freq[token] += 1

dict(sorted(term_dict.items())) #get all keys from term_dict and sort them
feature_id = 0 #defining constants
for key in term_dict:
    #wordfreq['feature_id'] = feature_id #adding feature id to the word frequency dictionary
    term_dict[key] = feature_id
    feature_id += 1

#define the class dictionary for each folder
class_dict = {"comp.graphics":"0", "comp.os.ms-windows.misc":"0", "comp.sys.ibm.pc.hardware":"0", "comp.sys.mac.hardware":"0", "comp.windows.x":"0",
            "rec.autos":"1", "rec.motorcycles":"1", "rec.sport.baseball":"1", "rec.sport.hockey":"1", 
            "sci.crypt":"2", "sci.electronics":"2", "sci.med":"2", "sci.space":"2",
            "misc.forsale":"3",
            "talk.politics.misc":"4", "talk.politics.guns":"4", "talk.politics.mideast":"4",
            "talk.religion.misc":"5", "alt.atheism":"5", "soc.religion.christian":"5"}

# generating the training data file(s)
class_id =''
feature_value =''
#trainingdatafile = class_id +" "+id+":"+feature_value
trainingdatafile = ''
for x in range(len(alldocuments)):
   for name in files:
       if directories in class_dict.values():
           class_id = class_dict.keys
           print("inside class_dict")
       for token in name:
           if token in term_dict.values():
               id = term_dict.keys
               feature_value = term_dict.values
               print("inside term_dict")
               trainingdatafile = ("%i %i:%s" %(class_id, id, feature_value))

with open('/Users/nat2147/Anant/code/python/trainingdatafile.txt', 'w') as f:
    print(trainingdatafile, file=f)
    #generate other files with all the information ready

'''
iterate alldocuments:
for each document:
   find the class-id with class-dict["subdirectory"] of the document
   for each term in the last part:
   	 use term-dict[term] to find the feature id
   put all information to the output string 
   with open('/Users/nat2147/Anant/code/python/trainingdatafile.txt', 'a') as f:
       print(trainingdatafile, file=f)
   you can also generate other files with all the information ready

#one row per document
#<class_label> <feature_id>:<feature_value> 
trainingdatafile = {}
fid = {}
#generate the training data set
for (root, directories, files) in os.walk(path):
    for d in directories:
        for k, v in class_dict.items():
            if d in v:
                trainingdatafile.update(k)
        for name in files:
            f1 = open(root+'/'+name, 'r', errors='ignore')
            content1 = f1.readlines() #reads all the lines of a file
            for sentence in content1:
                token1 = nltk.word_tokenize(sentence) #tokenizes the sentence
                for k, v in wordfreq.items():
                    if token1 in v:
                        trainingdatafile.update(k)

                #for token in token1:
                #    if token in wordfreq:
                #        fid =  wordfreq.keys()
            trainingdatafile.update(fid) #write the file output as a row to the dictionary
    
    with open('/Users/nat2147/Anant/code/python/trainingdatafile.txt', 'a') as f:
        print(trainingdatafile, file=f)

        


####
Code for making the bag of words into boolean model
####
sentence_vectors = []
for sentence in content:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in wordfreq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
'''