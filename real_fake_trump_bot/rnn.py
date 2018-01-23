import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, cudnn_rnn
from sklearn.model_selection import train_test_split
from real_fake_trump_bot.preprocessor import Preprocessor
import pdb

class RNN(Preprocessor):

    def __init__(
            self,
            tweets=None,
            learning_rate=0.0001,
            iterations=1000,
            batch_size=128,
            inspect_rate=50,
            hidden_layers=(16,8),
            tweet_size=140,
            dropout=0.3
    ):
        # initialize Preprocessor with DataFrame of tweets / twitter data
        super().__init__(tweets=tweets)

        # initialize hyperparameters and rnn attributes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.inspect_rate = inspect_rate
        self.hidden_layers = hidden_layers
        self.tweet_size = tweet_size
        self.dropout = dropout

        # data ready for training
        self.valid_data = False


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
        self.data_prepared = True

    def declare_model(self, skip_preprocess=False):
        if not skip_preprocess:
            assert self.data_prepared, "Preprocess your data first"

        # isolate training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.twitter_data.one_hot_encoded.values)

        # initialize placeholders for input and output values
        self.X = tf.placeholder(tf.float32, [None, self.iterations, self.tweet_size])
        self.y = tf.placeholder(tf.float32, [None, len(list(self.label_encoder.classes_))])
        self.input = tf.unstack(self.x, self.timesteps, 1)

        # initialize weights and biases


        # define lstm layer


        # prediction
        self.pred = None

        # loss function
        self.loss = None

        # optimizer
        self.opt = None

        # evaluate prediction


        # check accuracy
        self.accuracy = None


    def train(self):
        initializer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initializer)
            i = 1
            train_data = zip(self.x_train, self.y_train)
            while i < self.iterations:
                batch_x, batch_y = train_data.next_batch(batch_size=self.batch_size)
                batch_x = batch_x.reshape((self.batch_size, self.iterations, self.tweet_size))
                sess.run(self.opt, feed_dict={self.X: batch_x, self.y: batch_y})
                if i % 50 == 0:
                    acc = sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    print(i, acc, loss)
                i += 1

    def test(self):
        pass

    def generate_tweet(self):
        pass

    def decode_tweet(self, generated_tweet):
        return self.label_encoder.inverse_transform(np.argmax(generated_tweet, axis=0))

    def print_tweet(self, decoded_tweet):
        return ''.join(decoded_tweet)

    def inspect(self):
        pass



rnn = RNN()
rnn.preprocess_tweets()
pdb.set_trace()