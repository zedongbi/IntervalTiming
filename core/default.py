import numpy as np
from . import task


def input_output_n(rule_name):
    # basic timing tasks
    if rule_name == 'interval_production':
        return 2, 1
    elif rule_name == 'interval_comparison':
        return 2, 2
    # combined timing tasks
    elif rule_name == 'timed_spatial_reproduction':
        return 32+2, 32
    elif rule_name == 'timed_spatial_reproduction_broad_tuning':
        return 32+12+2, 32+12
    elif rule_name == 'timed_decision_making':
        return 2+1, 2
    # non-timing tasks
    elif rule_name == 'spatial_reproduction':
        return 32+1, 32
    elif rule_name == 'spatial_reproduction_broad_tuning':
        return 32+12+1, 32+12
    elif rule_name == 'spatial_reproduction_variable_delay':
        return 32+1, 32

    elif rule_name == 'spatial_comparison':
        return 32+32, 2
    elif rule_name == 'spatial_comparison_broad_tuning':
        return 32+12+32+12, 2
    elif rule_name == 'spatial_comparison_variable_delay':
        return 32+32, 2

    elif rule_name == 'spatial_change_detection':
        return 32+32, 2
    elif rule_name == 'spatial_change_detection_broad_tuning':
        return 32+12+32+12, 2
    elif rule_name == 'spatial_change_detection_variable_delay':
        return 32+32, 2

    elif rule_name == 'decision_making':
        return 2, 2
    elif rule_name == 'decision_making_variable_delay':
        return 2, 2

    elif rule_name == 'ctx_decision_making':
        return 2+2, 2
    elif rule_name == 'ctx_decision_making_variable_delay':
        return 2+2, 2


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
        # Type of activation functions: relu, softplus
        'activation': 'softplus',
        # Time constant (ms)
        'tau': 20,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3,#0.25,#0.27,#0.3,
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
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # learning rate
        'learning_rate': 0.0005,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
    }

    return hp
