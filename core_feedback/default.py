import numpy as np
from . import task


def input_output_n(rule_name):
    '''
    :param rule_name:
    :return: (total number of input, number of input with feedback, output)
    '''
    if rule_name == 'spatial_reproduction':
        return 32+1, 1, 32
    elif rule_name == 'spatial_comparison':
        return 32+32, 32, 2
    elif rule_name == 'spatial_change_detection':
        return 32+32, 32, 2
    elif rule_name == 'ctx_decision_making':
        return 2 + 2, 2, 2


def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    n_input, n_input_feedback, n_output = input_output_n(rule_name)

    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(1000000)
    else:
        seed = random_seed

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
        'tau': 20,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # a default weak regularization prevents instability
        'l1_firing_rate': 0,
        # l2 regularization on activity
        'l2_firing_rate': 0,
        # l1 regularization on weight
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # l2 regularization on feedback current
        'l2_feedback_current': 1e-4,
        # number of input units
        'n_input': n_input,
        # number of input units with feedback
        'n_input_feedback': n_input_feedback,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # learning rate
        'learning_rate': 0.0005,
        # first input index for rule units
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
    }

    return hp
