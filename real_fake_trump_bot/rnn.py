import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from real_fake_trump_bot.preprocessor import Preprocessor
import pdb

class RNN(Preprocessor):

    def __init__(self, tweets=None, learning_rate=0.0001, inspect_rate=50, hidden_layers=(16,8)):
        super().__init__(tweets=tweets)
        self.learning_rate = learning_rate
        self.inspect_rate = inspect_rate
        self.hidden_layers = hidden_layers


    def preprocess_tweets(self):
        print('Sanitizing tweets')
        self.sanitize_tweets(remove=('links'))

        print('Populating lexicon with utf-8 characters')
        self.populate_char_lexicon()

        print('Fitting label encoder')
        self.fit_label_encoder()

        print('Fitting one hot encoder')
        self.fit_one_hot_encoder()

        print('Labelling tweets')
        self.label_encode()

        print('One hot encoding tweets')
        self.one_hot_encode()

        print("You're all set to train, woot!")

    def train(self):
        pass

    def test(self):
        pass



rnn = RNN()
rnn.preprocess_tweets()
pdb.set_trace()