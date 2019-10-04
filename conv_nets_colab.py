#conv_nets.py
import torch
import os
from nltk import sent_tokenize,word_tokenize
from collections import Counter
import time
from string import punctuation
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , TensorDataset
from torch import nn
from define_model import process_model
#from string import maketrans
#Global variables
punctuation = punctuation + "/" + ">" + "<" + "!" + "+"

class sentiment:

    def __init__(self):
        self.labels = [] #List containing the labels associated with the reviews
        self.text = []    #List of lists where each list contains an individual review
        self.entire_text = " " #Entire texts
        self.no_punct = " "   #Entire text with no punctuation
        self.vocab_to_int = {}   #Vocab to int mapping where the most common word has the least integer mapping
        self.reviews_split = []
        self.reviews_int = []
        self.encoded_labels = []
        self.seq_length = 500
        self.review_len = []
        #self.train_x = []
        #self.train_y = []
        #self.test_x = []
        #self.test_y = []
        #self.valid_x = []
        #self.valid_y = []
        #self.features is a data member with reviews_len as rows and seq_length columns
        #self.x
        #self.words gives us all the individual words in the training data

    def parse_dataset(self):
        flag_break = 0
        count = 0
        imdb_dir = "/content/sentiment_analyze/aclImdb/"
        train_dir = os.path.join(imdb_dir,'train')
        print("train_dir:",train_dir)

        self.labels = []
        self.text = []
        #Parsing through the labels
        for label_type in ['neg','pos']:
            dir_name = os.path.join(train_dir,label_type)
            print("dir_name:",dir_name)
            for fname in os.listdir(dir_name):
                if(fname[-4:] == ".txt"):
                    f = open(os.path.join(dir_name,fname))
                    self.text.append([f.read()])
                    f.close()

                    if(label_type == "neg"):
                        self.labels.append(0)
                    elif(label_type == "pos"):
                        self.labels.append(1)

                count = count + 1  #Use this for debugging...
                #inp = input("Waiting for input..")
                # if(count == 3):

                    #   print("text:",text)
                    #   print("labels:",labels)
                        
                if(count == 20000):
                    flag_break = 1
                    break
            if(flag_break == 1):
                flag = 0
                break

        #Concatenating the list into an entire string of text
        print("Started concatenation...")
        for i in range(0,len(self.text)):
            self.entire_text = self.entire_text + ' '.join(self.text[i]) + "\n"
                
        print("Completed concatenating into an entire string of text")
        #print("self.entire_text:",self.entire_text)
        #print("self.text:",self.text)
        #print("self.labels:",self.labels)
        #print("len(labels):",len(self.labels))
        #print("len(text):",len(self.text))
    

    def tokenize(self):
        #self.reviews_split = self.entire_text.split('\n')
        #print("self.reviews_split:",self.reviews_split)
        self.entire_text = self.entire_text.lower()
        #print("self.entire_text:",self.entire_text)
        #inp = input("Waiting for input...")

#         for char in self.entire_text:
#             if char not in punctuation:
#                 self.no_punct = self.no_punct + char

        self.no_punct = self.entire_text.translate(str.maketrans('', '', punctuation))

        print("Completed punctuation removal")
        self.reviews_split = self.no_punct.split("\n")
        self.reviews_split.pop()
        print("Number of reviews:",len(self.reviews_split))

 
        self.words = self.no_punct.split()
        #print("words:",words)

        #Counting all of the words
        count_words = Counter(self.words)
        
        total_words = len(self.words)
        sorted_words = count_words.most_common(total_words)

        #print("count_words:",count_words)
        #print("sorted_words:",sorted_words)

        #Creating a vocab to int mapping
        self.vocab_to_int = {w:i+1 for i,(w,c) in enumerate(sorted_words)} #Here the most common word gets the lowest value and the least common words get the highest value
        print("Completed vocab to int mapping")
        #print("vocab_to_int:",self.vocab_to_int)
       
        #Encoding the words

        self.reviews_int = []
        for review in self.reviews_split:
            r = [self.vocab_to_int[w] for w in review.split()]
            self.reviews_int.append(r)
        print("Completed creating self.reviews_int")
        #print("self.reviews_int:",self.reviews_int)

        #Encoding the labels
        self.encoded_labels = [1 if label == "positive" else 0 for label in self.labels]
        self.encoded_labels = np.array(self.encoded_labels)
        print(type(self.encoded_labels))
        print("Completed encoding labels")
        #print("self.encoded_labels:",self.encoded_labels)

        
    def analyze_data(self):
        
        #Visualizing the length of each review
        self.review_len = [len(review) for review in self.reviews_int]
        #print("review_len:",review_len)
        #plt.plot(review_len)
        #plt.xlabel("Review number")
        #plt.ylabel("Review length")
        #plt.show()

        #print(pd.Series(review_len).describe())

        #Removing Outliers
        self.reviews_int = [self.reviews_int[i] for i,l in enumerate(self.review_len) if l>0]
        self.encoded_labels = [self.encoded_labels[i] for i,l in enumerate(self.review_len) if l>0]

        self.encoded_labels = np.array(self.encoded_labels)
        print("len(self.reviews_int):",len(self.reviews_int))
        print("len(self.encoded_labels):",len(self.encoded_labels))

    def prepare_data(self):
        #Pruning all reviews to length = seq_length
        self.features = np.zeros([len(self.reviews_int) , (self.seq_length)], dtype=int)
        print("self.features.shape:",self.features.shape)

        for i,review_int in enumerate(self.reviews_int):
            if(len(review_int) >= self.seq_length):
                temp = review_int[0:self.seq_length]
            
            elif(len(review_int) < self.seq_length):
                #zeros = np.zeros(self.seq_length - len(review_int))
                zeros = [0] * (self.seq_length - len(review_int))
                temp = review_int + zeros

            #print(np.array(temp).shape)
            self.features[i,:] = np.array(temp)  #So each row of our np array would contain the encoding for each particular review
        
        print(int(len(self.review_len) * 0.8))
        #Splitting into training,testing and validation sets
        self.train_x = self.features[0:int(len(self.review_len) * 0.8),:]
        self.train_y = self.encoded_labels[0:int(len(self.review_len) * 0.8)]

        remaining_x = self.features[int(len(self.review_len)*0.8):,:]
        remaining_y = self.encoded_labels[int(len(self.review_len)*0.8):]

        self.test_x = remaining_x[0:int(len(remaining_x)*0.5),:]
        self.test_y = remaining_y[0:int(len(remaining_y)*0.5)]
 
        self.valid_x = remaining_x[int(len(remaining_x)*0.5):,:]
        self.valid_y = remaining_y[int(len(remaining_y)*0.5):]


    def datasets(self):
        self.train_data = TensorDataset(torch.from_numpy(self.train_x),torch.from_numpy(self.train_y))
        self.valid_data = TensorDataset(torch.from_numpy(self.valid_x),torch.from_numpy(self.valid_y))
        self.test_data = TensorDataset(torch.from_numpy(self.test_x),torch.from_numpy(self.test_y))

        self.train_loader = DataLoader(self.train_data , shuffle = True , batch_size = 50)
        self.test_loader = DataLoader(self.test_data , shuffle = True , batch_size = 50)
        self.valid_loader = DataLoader(self.valid_data , shuffle = True , batch_size = 50)

        process_model(self.words,self.train_loader,self.test_loader,self.valid_loader)

    def visualize(self):
        dataiter = iter(self.train_loader)
        sample_x , sample_y = dataiter.next()
        self.x = sample_x.type(torch.LongTensor)

        print("sample_x:",sample_x.size())
        print("sample_y:",sample_y.size())


if __name__ == "__main__":
    print("punctuation:",punctuation)
    s = sentiment()
    s.parse_dataset()
    s.tokenize()
    s.analyze_data()
    s.prepare_data()
    s.datasets()
    #s.visualize()