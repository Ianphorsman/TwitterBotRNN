import numpy as np
import tensorflow as tf
from functools import wraps
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
            fixed_tweet_size=150,
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
        self.fixed_tweet_size = fixed_tweet_size
        self.dropout = dropout

        # data ready for training
        self.data_prepared = False


    def preprocess_tweets(self):
        print('Sanitizing tweets')
        self.sanitize_tweets(remove=('links'), strict_size=(False, self.fixed_tweet_size))

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



    def declare_model(self, dropout=0.3, skip_preprocess=False):
        if not skip_preprocess:
            assert self.data_prepared, "Preprocess your data first"

        # n classes (total number of unique characters)
        self.vocab_size = len(list(self.label_encoder.classes_))

        # number of tweets to process
        self.num_tweets = self.twitter_data.one_hot_encoding.values.shape[0]

        # initialize placeholders for input and output values
        self.X = tf.placeholder(tf.int32, [self.batch_size, self.fixed_tweet_size - 1, self.vocab_size], name='X')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.vocab_size], name='y')
        #pdb.set_trace()
        #self.input = tf.unstack(self.X, self.fixed_tweet_size, 1)

        # initialize weights and biases
        self.out_weights = tf.Variable(tf.random_normal([self.fixed_tweet_size, self.vocab_size]), name='out_weights')
        self.out_bias = tf.Variable(tf.random_normal([self.vocab_size]), name='out_bias')

        # generate model layers
        rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size), output_keep_prob=self.dropout) for size in self.model_shape]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        self.outputs, self.state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=self.X, sequence_length=[self.fixed_tweet_size-1], dtype=tf.float32)

        # prediction
        self.prediction = tf.matmul(self.outputs[-1], self.out_weights) + self.out_bias

        # loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))

        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # evaluate prediction
        #self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))

        # check accuracy
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def train(self):
        initializer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initializer)
            i = 1
            train_X = np.concatenate([[tweet] for tweet in self.twitter_data.one_hot_encoding.values], axis=0)
            train_y = np.roll(train_X, -1, axis=1)
            pdb.set_trace()
            #train_X = np.concatenate(self.twitter_data.one_hot_encoding.values)
            #train_y = np.roll(train_X, -1, axis=0)
            batch = self.next_batch()
            while i < self.iterations:
                idx_start, idx_stop = next(batch)
                batch_x = train_X[idx_start:idx_stop]
                batch_y = train_y[idx_start:idx_stop]
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.y: batch_y})
                if i % 50 == 0:
                 #   acc = sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    print(i, loss)
                i += 1

    def next_batch(self):
        stop = self.num_tweets * self.iterations
        for idx in list(range(0, stop, self.batch_size)):
            yield idx % self.num_tweets, min((idx % self.num_tweets) + self.fixed_tweet_size, self.num_tweets)


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


'''
char_rnn = Char_RNN(iterations=100, batch_size=150, fixed_tweet_size=150, model_shape=(150, 150))
char_rnn.preprocess_tweets()
char_rnn.declare_model(dropout=0.4)
print(char_rnn.y.shape)
print(char_rnn.X.shape)
char_rnn.train()
'''
