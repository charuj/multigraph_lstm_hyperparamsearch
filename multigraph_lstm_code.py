import tensorflow as tf


def _init_(self, learning_rate, num_layers, hidden_units, batch_size, seq_len, num_features):
    self.graph= tf.Graph()
    with self.graph.as_default():
        self.x= tf.placeholder(tf.float32, shape=[None, seq_len, num_features]) # None is for batch_size
        print 'X', self.x # inputs
        self.y= tf.placeholder(tf.float32, shape=[None, seq_len, 1]) # targets

        weights_out= tf.get_variable('wout', shape=[hidden_units, seq_len], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None, dtype=tf.float32))

        biases_out= tf.get_variable('bout', shape=[seq_len], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None, dtype=tf.float32))

        cells=[]
        for _ in range(num_layers):
            cell= tf.contrib.rnn.LSTMCell(hidden_units)
            cells.append(cell)
        cell= tf.contrib.rnn.MultiRNNCell(cells)

        # Batch_size x time_steps x features
        print self.x.shape
        output, state= tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        self.prediction= tf.matmul(output[-1], weights_out)+biases_out
        print 'pred: ', self.prediction

        

