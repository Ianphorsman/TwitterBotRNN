import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from real_fake_trump_bot.preprocessor import Preprocessor

class RNN(Preprocessor):

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



bot = TrumpBot()
bot.sanitize_tweets(remove=('links'))
bot.extend_features(tweet_tokenize=True)
bot.populate_char_lexicon('Clean_Tweets')
bot.fit_label_encoder()
bot.fit_one_hot_encoder()
bot.label_encode()
bot.one_hot_encode()
bot.remove_raw_tweets()
print(bot.twitter_data.head())