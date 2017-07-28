import numpy as np


num_searches=5 #aribtrarily chosen


def hyperparam_search(num_searches):
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
        hyperparam_dict[dict_name]['num_layer']=layer

        #Learning Rate
        learning_rate= np.random.choice(np.logspace(start=-5, stop=0, num=num_searches))
        #Choose between 0.00001 and 1
        rounded_learning_rate = np.around(learning_rate, decimals=5)
        hyperparam_dict[dict_name]['learning_rate']= rounded_learning_rate

        #Batch Size
        batch_size= np.random.randint(low=64, high=200)
        hyperparam_dict[dict_name]['batch_size']= batch_size

    return hyperparam_dict

dict= hyperparam_search(num_searches)
print dict
