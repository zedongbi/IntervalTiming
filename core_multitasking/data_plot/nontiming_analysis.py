import numpy as np
from matplotlib import pyplot as plt

import os

from .. import run

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)

root_directory = os.path.join(os.getcwd(),'core_multitasking')

transient_period = 20


def temporal_variance_spatial_reproduction(serial_idxes, noise_on=False):

    spatial_reproduction_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'spatial_reproduction/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32.-6., 2.)
        batch_size = len(ring_centers)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        # shape: time, batch_for_direction, neuron
        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        spatial_reproduction_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return spatial_reproduction_t


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
        decision_making_t.append(np.var(firing_rate_list_t) / np.var(firing_rate_list_centered))

    return decision_making_t
