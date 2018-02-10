import numpy as np
import tensorflow as tf
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
        self.dropout = dropout

        # data ready for training
        self.data_prepared = False


    def preprocess_tweets(self):
        print('Sanitizing tweets')
        self.sanitize_tweets(remove=('links'), strict_size=(True, self.fixed_tweet_size))

        print('Populating lexicon')
        self.populate_char_lexicon()

        print('Creating lookup dictionary')
        self.create_lookup_dictionary()

        print("You're all set to train, woot!")
        self.data_prepared = True

    def declare_model(self, dropout=0.3, skip_preprocess=False):
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

        self.testX = tf.placeholder(tf.int32, [None, 1], name='testX')
        test_one_hot_encoded = tf.one_hot(self.testX, self.vocab_size)

        #rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True), output_keep_prob=self.dropout) for size in self.model_shape]
        #multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        #self.outputs, self.state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=self.X,
         #                                            sequence_length=[self.fixed_tweet_size], dtype=tf.float32)
        self.rnn_cell = rnn.LSTMCell(num_units=250)
        self.lstm_init_value = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_cell, one_hot_encoded, initial_state=self.lstm_init_value, dtype=tf.float32)
        self.test_outputs, self.test_states = tf.nn.dynamic_rnn(self.rnn_cell, test_one_hot_encoded, initial_state=self.lstm_init_value, dtype=tf.float32)
        #self.outputs = tf.squeeze(self.outputs, [0]) # removes all size 0 dimensions

        self.logits = tf.layers.dense(self.outputs, self.vocab_size, None, True, tf.orthogonal_initializer(), name='dense')
        self.test_logits = tf.layers.dense(self.test_outputs, self.vocab_size, None, True, tf.orthogonal_initializer(), name='testdense') # might not reuse

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self):
        initializer = tf.global_variables_initializer()
        self.train_data = [self.char_to_index[char] for char in "".join(self.twitter_data.Clean_Tweets.values)]
        tweets = self.next_tweet()
        self.train_x = np.atleast_2d([self.train_data[idx_start:idx_stop][:-1] for idx_start, idx_stop in tweets][:-1])
        tweets = self.next_tweet()
        self.train_y = np.atleast_2d(np.roll([self.train_data[idx_start:idx_stop][:-1] for idx_start, idx_stop in tweets], shift=-1)[:-1])
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
                #pdb.set_trace()
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.y: batch_y})
                if i % self.inspect_rate == 0:
                    #acc = sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss, outputs, states, logits = sess.run([self.loss, self.outputs, self.states, self.logits], feed_dict={self.X: batch_x, self.y: batch_y})
                    print(i, loss, logits.shape)
                    #if i >= 200:
                    #    print(self.generate_tweet(sess))
                i += 1
        #self.generate_tweet()


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

    def generate_tweet(self, sess, start='stop countr'):
        predicted_tweet = start
        predicted_tweet_2 = start
        state = sess.run(self.rnn_cell.zero_state(1, dtype=tf.float32))
        #pdb.set_trace()
        batch_x = np.atleast_2d(list([self.char_to_index[char] for char in start]))
        states, logits = sess.run([self.test_states, self.test_logits], feed_dict={self.testX: batch_x, self.lstm_init_value: state})
        softmax = None
        for i in range(140 - len(start)):
            softmax = tf.nn.softmax(logits)
            output = tf.argmax(softmax, axis=1)
            batch_x = np.atleast_2d(list([self.char_to_index[char] for char in predicted_tweet]))
            states, logits = sess.run([self.test_states, self.test_logits], feed_dict={self.testX: batch_x, self.lstm_init_value: states})


        return predicted_tweet

    def print_tweet(self, decoded_tweet):
        return ''.join(decoded_tweet)

    def inspect(self):
        pass



char_rnn = CharRNN(iterations=650, inspect_rate=50, learning_rate=0.01, batch_size=750, fixed_tweet_size=150, model_shape=(150, 150))
char_rnn.preprocess_tweets()
char_rnn.declare_model(dropout=0.4)
print(char_rnn.y.shape)
print(char_rnn.X.shape)
char_rnn.train()
