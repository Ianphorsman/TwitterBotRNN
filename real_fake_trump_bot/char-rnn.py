import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib import rnn, cudnn_rnn
from real_fake_trump_bot.tf_preprocessor import TFTweetPreprocessor
import pdb


class CharRNN(TFTweetPreprocessor):

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
        self.sequence_length = self.fixed_tweet_size
        self.dropout = tf.placeholder_with_default(dropout, shape=())

        # data ready for training
        self.data_prepared = False


    def preprocess_tweets(self):
        print('Sanitizing tweets')
        self.sanitize_tweets(remove=('links', 'nan'), tags=('end'), strict_size=(False, self.fixed_tweet_size))

        print('Populating lexicon')
        self.populate_char_lexicon()

        print('Creating lookup dictionary')
        self.create_lookup_dictionary()

        print("You're all set to train, woot!")
        self.data_prepared = True

    def declare_model(self, skip_preprocess=False):
        if not skip_preprocess:
            assert self.data_prepared, "Preprocess your data first"

        # n classes (total number of unique characters)
        self.vocab_size = len(list(self.lexicon.keys()))

        # number of tweets to process
        self.num_tweets = self.twitter_data.Clean_Tweets.values.shape[0]

        self.X = tf.placeholder(tf.int32, [self.batch_size, self.fixed_tweet_size - 1], name='X')
        one_hot_encoded = tf.one_hot(self.X, self.vocab_size)
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.fixed_tweet_size - 1], name='y')
        labels = tf.one_hot(self.y, self.vocab_size)

        self.testX = tf.placeholder(tf.int32, [1, None], name='testX')
        test_one_hot_encoded = tf.one_hot(self.testX, self.vocab_size)

        rnn_layers = [rnn.LayerNormBasicLSTMCell(size, forget_bias=1.0) for size in self.model_shape]
        self.multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        self.lstm_init_value = self.multi_rnn_cell.zero_state(self.batch_size, tf.float32)
        self.test_lstm_init_value = self.multi_rnn_cell.zero_state(1, tf.float32)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_rnn_cell, one_hot_encoded, initial_state=self.lstm_init_value, dtype=tf.float32)
        self.test_outputs, self.test_states = tf.nn.dynamic_rnn(self.multi_rnn_cell, test_one_hot_encoded, initial_state=self.test_lstm_init_value, dtype=tf.float32)

        self.flat_outputs = tf.reshape(self.outputs, [-1, self.model_shape[-1]])
        self.test_flat_outputs = tf.reshape(self.test_outputs, [-1, self.model_shape[-1]])

        self.logits = tf.layers.dense(self.flat_outputs, self.vocab_size, None, True, tf.orthogonal_initializer(), name='dense')
        self.test_logits = tf.layers.dense(self.test_flat_outputs, self.vocab_size, None, True, tf.orthogonal_initializer(), name='testdense') # might not reuse


        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(labels, [-1, self.vocab_size])))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self):
        initializer = tf.global_variables_initializer()
        text = "".join(self.twitter_data.Clean_Tweets.values)
        text = text[:-(len(text) % 150)]
        #pdb.set_trace()
        self.train_data = [self.char_to_index[char] for char in text]
        tweets = self.next_tweet()
        self.train_x = np.atleast_2d([self.train_data[idx_start:idx_stop][:-1] for idx_start, idx_stop in tweets][:-1])
        tweets = self.next_tweet()
        self.train_y = np.atleast_2d(np.roll([self.train_data[idx_start:idx_stop][:-1] for idx_start, idx_stop in tweets], shift=-1)[:-1])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(initializer)
            #pdb.set_trace()
            i = 1
            batch = self.next_batch()
            while i < self.iterations:
                idx_start, idx_stop, batch_size = next(batch)
                batch_x = self.train_x[idx_start:idx_stop]
                batch_y = self.train_y[idx_start:idx_stop]
                if not batch_size == self.batch_size:
                    batch_x = np.concatenate([batch_x, self.train_x[0:(self.batch_size - batch_size)]])
                    batch_y = np.concatenate([batch_y, self.train_y[0:(self.batch_size - batch_size)]])
                _, logits, outputs = sess.run([self.optimizer, self.logits, self.outputs], feed_dict={self.X: batch_x, self.y: batch_y})
                #pdb.set_trace()
                if i % self.inspect_rate == 0:
                    loss, outputs, states, logits = sess.run([self.loss, self.outputs, self.states, self.logits], feed_dict={self.X: batch_x, self.y: batch_y})
                    print(i, loss, logits.shape)
                    if i >= 400:
                        self.gen_tweet(sess)
                i += 1
            saver.save(sess, save_path=os.path.abspath(os.path.join(os.getcwd(), 'modelv1')))
            #print(self.gen_tweet(sess, start='today we ex'))
            #print(self.gen_tweet(sess, start='th'))
            #print(self.gen_tweet(sess, start='build '))
            #print(self.gen_tweet(sess, start='crooke'))


    def next_tweet(self):
        stop = len(self.train_data)
        for idx in list(range(0, stop, self.fixed_tweet_size)):
            yield idx % stop, min((idx % stop) + self.fixed_tweet_size, stop)

    def next_batch(self):
        stop = len(self.train_x) // self.fixed_tweet_size
        for idx in list(range(0, len(self.train_x) * self.iterations, self.batch_size)):
            idx_start = idx % stop
            idx_stop = min((idx % stop) + self.batch_size, stop)
            yield idx_start, idx_stop, idx_stop - idx_start


    def test(self):
        pass

    def gen_tweet(self, sess):
        tweet = self.twitter_data.Clean_Tweets[np.random.choice(1700)]
        start = tweet[:50]
        predicted_tweet = start
        state = sess.run(self.multi_rnn_cell.zero_state(1, dtype=tf.float32))
        for char in start[:-1]:
            state = sess.run(self.test_states, feed_dict={self.testX: np.atleast_2d([self.char_to_index[char]]), self.test_lstm_init_value: state})

        #pdb.set_trace() # check state size
        for _ in range(140 - len(start)):
            feed = {self.testX: np.atleast_2d([self.char_to_index[start[-1]]])}#, self.test_lstm_init_value: state}
            state, logits = sess.run([self.test_states, self.test_logits], feed_dict=feed)
            #pdb.set_trace()
            #probs = tf.nn.softmax(tf.squeeze(logits))
            probs = tf.nn.softmax(logits)
            chars = self.probs_to_chars(probs.eval()[0], just_chars=True)
            chars_probs = self.probs_to_chars(probs.eval()[0])
            predicted_tweet += chars[0]
            #pdb.set_trace()

        print(tweet)
        print(predicted_tweet)

    def generate_tweet(self, sess, start='today we expr'):
        predicted_tweet = start
        predicted_tweet_2 = start
        self.dropout = 1.0
        initial_state = sess.run(self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32))
        #pdb.set_trace()
        batch_x = np.atleast_2d(list([self.char_to_index[char] for char in start]))
        states, logits = sess.run([self.test_states, self.test_logits], feed_dict={self.testX: batch_x, self.lstm_init_value: initial_state})
        softmax = None
        for i in range(140 - len(start)):
            softmax = tf.nn.softmax(logits[-1])
            output = tf.argmax(softmax, axis=1)
            batch_x = np.atleast_2d(list([self.char_to_index[char] for char in predicted_tweet]))
            states, logits = sess.run([self.test_states, self.test_logits], feed_dict={self.testX: batch_x, self.lstm_init_value: states})


        return predicted_tweet

    def probs_to_chars(self, probs, just_chars=False):
        chars = sorted([(idx, self.index_to_char[i]) for i, idx in enumerate(probs)], key=lambda tup: tup[0], reverse=True)
        if just_chars:
            return ''.join(list(map(lambda tup: tup[1], chars)))
        return chars

    def print_tweet(self, decoded_tweet):
        return ''.join(decoded_tweet)

    def inspect(self):
        pass



char_rnn = CharRNN(iterations=800, dropout=1.0, inspect_rate=25, learning_rate=0.01, batch_size=1, fixed_tweet_size=150, model_shape=(250,))
char_rnn.preprocess_tweets()
char_rnn.declare_model()
print(char_rnn.y.shape)
print(char_rnn.X.shape)
char_rnn.train()
