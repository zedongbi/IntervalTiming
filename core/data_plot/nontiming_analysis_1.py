from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

import os


from .. import run

'''
from __future__ import division

import torch
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import optimize
from scipy import signal
import os
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.stats import ortho_group
from collections import defaultdict
from numba import jit
from dPCA import dPCA

from .. import task
from .. import default
from .. import run
from .analysis import PCASpace
from .analysis import ScalingSpace
from .. import tools
'''

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

root_directory = os.path.join(os.getcwd(),'core')

transient_period = 20


def temporal_spatial_variance_decision_making(serial_idxes, noise_on=False):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()

    decision_making_t = list()
    decision_making_s = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'decision_making/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='decision_making', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c)

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate_list = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)

        temp = np.zeros((firing_rate_list.shape[0], 2, firing_rate_list.shape[2]))
        temp[:, 0, :] = np.mean(firing_rate_list[:, 0:3, :], axis=1)
        temp[:, 1, :] = np.mean(firing_rate_list[:, 3:, :], axis=1)
        firing_rate_list = temp

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)

        decision_making_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))
        decision_making_s.append(np.var(firing_rate_list_s)/np.var(firing_rate_list_centered))

    return decision_making_t, decision_making_s


def temporal_spatial_variance_timed_decision_making(serial_idxes, noise_on=False):
    #transient_period = 20

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()
    dly_intervals = np.array([1600.] * batch_size)
    prod_intervals = np.array([1200.] * batch_size)

    timed_decision_making_t = list()
    timed_decision_making_s = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/' + 'timed_decision_making/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_decision_making', is_cuda=False,
                               noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gamma_bar=gamma_bar, c=c)

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate_list = np.concatenate(list(
            firing_rate_binder[stim1_off[i] + transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in
            range(0, batch_size)), axis=1)

        temp = np.zeros((firing_rate_list.shape[0], 2, firing_rate_list.shape[2]))
        temp[:, 0, :] = np.mean(firing_rate_list[:, 0:3, :], axis=1)
        temp[:, 1, :] = np.mean(firing_rate_list[:, 3:, :], axis=1)
        firing_rate_list = temp

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)

        timed_decision_making_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))
        timed_decision_making_s.append(np.var(firing_rate_list_s)/np.var(firing_rate_list_centered))

    return timed_decision_making_t, timed_decision_making_s


def temporal_variance_decision_making(serial_idxes, noise_on=False):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()

    decision_making_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'decision_making/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='decision_making', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c)

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)
        # shape: neuron, time, c,
        firing_rate_list = firing_rate.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        decision_making_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return decision_making_t


def temporal_variance_ctx_decision_making(serial_idxes, noise_on=False):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()

    choice = np.array([1.] * batch_size).flatten()
    ctx_decision_making_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'ctx_decision_making/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='ctx_decision_making', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c, choice=choice)

        epoch = 'interval'

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)
        # shape: neuron, time, c,
        firing_rate_list = firing_rate.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        ctx_decision_making_t.append(np.var(firing_rate_list_t) / np.var(firing_rate_list_centered))

    return ctx_decision_making_t


def temporal_variance_decision_making_variable_delay(serial_idxes, noise_on=False):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()

    decision_making_no_timing_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'decision_making_variable_delay/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='decision_making_variable_delay', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c)

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)
        # shape: neuron, time, c,
        firing_rate_list = firing_rate.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        decision_making_no_timing_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return decision_making_no_timing_t


def temporal_variance_ctx_decision_making_variable_delay(serial_idxes, noise_on=False):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()

    choice = np.array([1.] * batch_size).flatten()
    decision_making_ctx_no_timing_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'ctx_decision_making_variable_delay/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='ctx_decision_making_variable_delay', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c, choice=choice)

        epoch = 'interval'

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        # shape: (time, c, neuron)
        firing_rate = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)
        # shape: neuron, time, c,
        firing_rate_list = firing_rate.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        decision_making_ctx_no_timing_t.append(np.var(firing_rate_list_t) / np.var(firing_rate_list_centered))

    return decision_making_ctx_no_timing_t
