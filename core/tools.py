"""Utility functions."""

import os
import errno
import json
import pickle
import numpy as np
import torch

from . import default


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        print(fname)
        hp = default.get_default_hp()

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['seed'] = np.random.randint(0, 1000000)
    hp['rng'] = np.random.RandomState(hp['seed'])
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_log(log, log_name='log.json'):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, log_name)
    with open(fname, 'w') as f:
        json.dump(log, f)


def load_log(model_dir, log_name='log.json'):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, log_name)
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def sequence_mask(lens):
    '''
    Input: lens: numpy array of integer

    Return sequence mask
    Example: if lens = [3, 5, 4]
    Then the return value will be
    tensor([[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]], dtype=torch.uint8)
    :param lens:
    :return:
    '''
    max_len = max(lens)
    # return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    return torch.t(torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(np.expand_dims(lens, 1), dtype=torch.float32))


