import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, cudnn_rnn
from real_fake_trump_bot.preprocessor import Preprocessor
import pdb

class Char_RNN(Preprocessor):

    def __init__(
            self,
            tweets=None,
            learning_rate=0.0001,
            iterations=1000,
            batch_size=128,
            inspect_rate=50,
            model_shape=(150,150),
            max_tweet_size=150,
            dropout=0.3
    ):
        # initialize Preprocessor with DataFrame of tweets / twitter data
        super().__init__(tweets=tweets)

        # initialize hyperparameters and rnn attributes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.inspect_rate = inspect_rate
        self.model_shape = model_shape
        self.max_tweet_size = max_tweet_size
        self.dropout = dropout

        # data ready for training
        self.data_prepared = False


    def preprocess_tweets(self):
        print('Sanitizing tweets')
        self.sanitize_tweets(remove=('links'), strict_size=(True, self.max_tweet_size))

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
        self.data_prepared = True
