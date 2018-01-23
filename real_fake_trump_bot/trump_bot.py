import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import _pickle as picklerick

class TrumpBot(object):

    def __init__(self):
        self.tweet_data = pd.read_csv('../data/Donald-Tweets.csv')


    def pca(self):
        pass

    def tSNE(self):
        pass

    def markov(self):
        pass

    def load(self, filename):
        setattr(self, filename, picklerick.load(open("data/{}.p".format(filename), 'rb')))

    def save(self, obj, filename):
        picklerick.dump(obj, open("data/{}.p".format(filename), 'wb'))

TrumpBot()