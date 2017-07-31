import tensorflow as tf
import numpy as np
import cPickle as pickle




def hyperparam_search(num_searches):
    '''
    :param num_searches: number of searches/experiments to run, where each experiment has a different set of hyperparameters
    :return: a dictionary of hyperparameters 
    '''
    hyperparam_dict={}

    for j in range(num_searches):
        ## Randomely select hyperparameters
        dict_name= 'parameter_set_%i' %j
        hyperparam_dict[dict_name]={}

        #Hidden Units
        num_hidden_units= np.random.choice(np.logspace(start=2.1, stop=3, num=num_searches))
        #chooses between 128 to 1000 hidden units, uniformly distributed along log space
        rounded_hidden_units= np.round(num_hidden_units)
        hyperparam_dict[dict_name]['hidden_units']= rounded_hidden_units

        #Number of Layers
        num_layers= np.random.uniform(low=1, high=3)
        layer= np.round(num_layers)
        hyperparam_dict[dict_name]['num_layers']=layer

        #Learning Rate
        learning_rate= np.random.choice(np.logspace(start=-5, stop=0, num=num_searches))
        #Choose between 0.00001 and 1
        rounded_learning_rate = np.around(learning_rate, decimals=5)
        hyperparam_dict[dict_name]['learning_rate']= rounded_learning_rate

        #Batch Size
        batch_size= np.random.randint(low=64, high=200)
        hyperparam_dict[dict_name]['batch_size']= batch_size

    return hyperparam_dict


#To build the graph
def Graph(self, learning_rate, num_layers, hidden_units, batch_size, seq_len, num_features):
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

        #Cost
        self.cost= tf.reduce_sum(tf.square(self.prediction - self.y))

        #Optimizer
        self.optimizer= tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost)


#To launch the graph
def launchG(self, train_inputs, train_targets, valid_inputs, valid_targets, num_epochs):
    with tf.Session(graph=self.graph) as sess:
        init= tf.global_variables_initializer()
        sess.run(init)

        valid_dict= {self.x: valid_inputs}
        valid_dict.update({self.y: valid_targets})

        for i in range(num_epochs):
            sess.run(self.optimizer, feed_dict={self.x: train_inputs, self.y:train_targets} )
            loss= sess.run(self.cost, feed_dict={self.x: train_inputs, self.y: train_targets})
            print loss

            valid_cost= sess.run(self.cost, feed_dict= valid_dict)
            pred= sess.run(self.prediction, feed_dict=valid_dict)
            print "Pred", pred
            print self.prediction.shape

            return valid_cost


## Running loop for experiments
for j in range(num_searches): # loop over the number of hyperparameter searches
    model_performance=[]
    keys= 'parameter_set_%i' %j
    learning_rate= hyperparam_dict[str(key)]['learning_rate'] # fetching the hyperparameters from the hyperparam dict
    num_layers= int(hyperparam_dict[str(key)]['num_layers'])
    hidden_units= int(hyperparam_dict[str(key)]['hidden_units'])
    batch_size= int(hyperparam_dict[str(key)]['batch_size'])

    k_fold_performance=[]
    for k in range(len(train_list)): #k times
        train_input= train_list[k]
        train_target= train_target_list[k]

        valid_input= valid_list[k]
        valid_target= valid_target_list[k]

        #Reshape data into [batch_size x seq_len x num_features]
        train_input, train_target= valid_reshape(train_input, train_target, seq_len)
        valid_input, valid_target= valid_reshape(valid_input, valid_target, 24)

        model= Graph(leaning_rate, num_layers, hidden_units, batch_size, seq_len, num_features)
        valid_cost_for_k= model.launchG(train_input, train_target, valid_input, valid_target, 40)
        k_fold_performance.append(valid_cost_for_k)


    model_avg_perf= np.mean(k_fold_performance)
    model_details= [model_avg_perf, learning_rate, num_layers, hidden_units, batch_size]
    model_performance.append(model_details)

    with open('hyperparameters_model_performance', 'a') as abc:
        np.savetxt(abc, model_performance, delimiter=",")




