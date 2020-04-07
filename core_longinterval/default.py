import numpy as np
from . import task


def input_output_n(rule_name):
    if rule_name == 'interval_production_long_interval':
        return 2, 1


def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    n_input, n_output = input_output_n(rule_name)

    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(1000000)
    else:
        seed = random_seed
    #seed = 321985
    hp = {
        'rule_name': rule_name,
        # batch size for training
        'batch_size_train': 64, #128,#64,
        # batch_size for testing
        'batch_size_test': 512,
        # Type of RNNs: RNN
        'rnn_type': 'RNN',
        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation runctions, relu, softplus, tanh, elu
        'activation': 'softplus',
        # Time constant (ms)
        'tau': 20, #20,  # 4
        # discretization time step (ms)
        'dt': 20, #5, #20,#10, #20,
        # discretization time step/time constant
        'alpha': 1,#0.5, #, # 1,  # 0.2
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.15,#0.3,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # a default weak regularization prevents instability
        'l1_firing_rate': 0, #1e-4, #0,
        # l2 regularization on activity
        'l2_firing_rate': 0,
        # l1 regularization on weight
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,  # 128,  #384, #256,  #128,  # 256,
        # learning rate
        'learning_rate': 0.0005, #0.0005,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
    }

    return hp
