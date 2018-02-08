import numpy as np
import tensorflow as tf
from collections import Counter
import pandas as pd
import string
import _pickle as picklerick
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk import pos_tag

class TFTweetPreprocessor(object):

    def __init__(self, tweets=None, tweet_text_col='Tweet_Text'):
        # get tweets from source file or from passed in DataFrame object
        self.tweet_text_col = tweet_text_col
        if tweets is None:
            self.twitter_data = pd.read_csv('../data/Donald-Tweets.csv')
        else:
            self.twitter_data = tweets

        self.tweets = self.twitter_data[self.tweet_text_col]
        self.validate_twitter_data()
        self.lexicon = Counter()
        self.tweet_tokenizer = TweetTokenizer()

    def validate_twitter_data(self):
        assert type(self.twitter_data) is pd.DataFrame, 'tweets need to be a pandas DataFrame object.'
        assert self.tweet_text_col in self.twitter_data.columns, "'{}' col name needs to be in tweets".format(self.tweet_text_col)
        tweets_dtype = list(set(map(lambda tweet: type(tweet), self.tweets.values)))
        assert len(tweets_dtype) == 1 and tweets_dtype[0] is str, 'All tweets must be strings'
        print("Twitter data looking good!")

    def load(self, filename):
        setattr(self, filename, picklerick.load(open("data/{}.p".format(filename), 'rb')))

    def save(self, obj, filename):
        picklerick.dump(obj, open("data/{}.p".format(filename), 'wb'))

    def sanitize_tweets(self, remove=('links', 'punctuation', 'nan'), strip=True, lower=True, strict_size=(True, 140)):
        self.twitter_data.loc[:, 'Clean_Tweets'] = self.tweets
        if 'links' in remove:
            self.twitter_data.Clean_Tweets = self.twitter_data.Clean_Tweets.str.replace('https?:\/\/.*[\r\n]*', '')
        if 'punctuation' in remove:
            self.twitter_data.Clean_Tweets = self.twitter_data.Clean_Tweets.str.translate(str.maketrans('', '', string.punctuation))
        if 'nan' in remove:
            self.twitter_data = self.twitter_data[self.twitter_data.Clean_Tweets.notnull()]
        if lower:
            self.twitter_data.Clean_Tweets = self.twitter_data.Clean_Tweets.str.lower()
        if strip:
            self.twitter_data.Clean_Tweets = self.twitter_data.Clean_Tweets.str.strip()
        if strict_size[0]:
            self.twitter_data.Clean_Tweets = self.twitter_data.Clean_Tweets.str.ljust(strict_size[1]).str[:strict_size[1]]

    def extend_features(
        self,
        num_chars=False,
        split_words=False,
        tweet_tokenize=False,
        pos=False,
        custom=None
    ):
        if num_chars:
            self.twitter_data.loc[:, 'num_chars'] = self.tweets.str.len()
        if split_words:
            self.twitter_data.loc[:, 'split_by_words'] = self.tweets.apply(lambda tweet: word_tokenize(tweet))
        if tweet_tokenize:
            self.twitter_data.loc[:, 'tokenized_tweets'] = self.twitter_data.Clean_Tweets.apply(
                lambda tweet: self.tweet_tokenizer.tokenize(tweet)
            )
        if pos:
            self.twitter_data.loc[:, 'part_of_speech'] = self.twitter_data.Clean_Tweets.apply(
                lambda tweet: pos_tag(word_tokenize(tweet))
            )
        if custom is not None:
            assert type(custom) is tuple, "Pass custom function as a tuple consisting of (feature_name, func)."
            self.twitter_data.loc[:, custom[0]] = self.tweets.apply(custom[1])

    def populate_word_lexicon(self, col_name='Clean_Tweets'):
        for tweet in self.twitter_data[col_name].values:
            self.lexicon.update(tweet)

    def populate_char_lexicon(self, col_name='Clean_Tweets'):
        text = "".join(self.twitter_data[col_name].values.ravel())
        for char in text:
            self.lexicon.update(char)

    def create_lookup_dictionary(self):
        lexicon = list(self.lexicon.keys())
        self.char_to_index = {char: i for (i, char) in enumerate(lexicon)}
        self.index_to_char = {i: char for (i, char) in enumerate(lexicon)}

    def remove_raw_tweets(self):
        self.tweets = self.twitter_data.Clean_Tweets
        self.twitter_data.pop(self.tweet_text_col)
