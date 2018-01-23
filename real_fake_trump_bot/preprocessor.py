import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import string
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk import pos_tag
import _pickle as picklerick

class Preprocessor(object):

    def __init__(self, tweets=None, tweet_text_col='Tweet_Text'):
        # get tweets from source file or from passed in DataFrame object
        self.tweet_text_col = tweet_text_col
        if tweets is None:
            self.twitter_data = pd.read_csv('../data/Donald-Tweets.csv')
        else:
            self.twitter_data = tweets

        self.tweets = self.twitter_data[self.tweet_text_col]
        self.validate_twitter_data()
        self.tweet_tokenizer = TweetTokenizer()
        self.lexicon = Counter()
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)

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

    def sanitize_tweets(self, remove=('links', 'punctuation', 'nan'), strip=True, lower=True):
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
        print(self.twitter_data['Clean_Tweets'].values[0:5])

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

    def fit_label_encoder(self):
        self.label_encoder.fit(list(self.lexicon.keys()))

    def fit_one_hot_encoder(self):
        self.one_hot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

    def label_encode(self):
        self.twitter_data.loc[:, 'label_encoding'] = self.twitter_data.Clean_Tweets.apply(
            lambda tweet: self.label_encoder.transform(np.array(list(tweet), dtype='<U32')).reshape(-1, 1)
        )

    def one_hot_encode(self, normalize=False):
        self.twitter_data.loc[:, 'one_hot_encoding'] = self.twitter_data.label_encoding.apply(
            lambda label_encoding: self.one_hot_encoder.transform(label_encoding).toarray() if label_encoding.shape[0] > 0 else None
        )

    def normalize(self, tweet):
        # didn't finish this feature, sorry
        pass

    def remove_raw_tweets(self):
        self.tweets = self.twitter_data.Clean_Tweets
        self.twitter_data.pop('Tweet_Text')