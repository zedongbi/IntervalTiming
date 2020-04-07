from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

import os

from .. import run


fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

root_directory = os.path.join(os.getcwd(),'core_feedback')

transient_period = 20


def temporal_variance_spatial_reproduction(serial_idxes, noise_on=False):

    spatial_reproduction_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'spatial_reproduction/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32.-6., 2.)
        batch_size = len(ring_centers)
        signal2_strength = 1.

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center=ring_centers, signal2_strength=signal2_strength)

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


def temporal_variance_spatial_comparison(serial_idxes, epoch='interval'):

    gaussian_center1 = np.arange(6., 32.-6, 2)
    gaussian_center2 = np.array([15.]*len(gaussian_center1))
    signal2_strength = 1.

    batch_size = len(gaussian_center1)

    spatial_comparison_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory + '/model/' + 'spatial_comparison/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_comparison', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2, signal2_strength=signal2_strength)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        spatial_comparison_t.append(np.var(firing_rate_list_t) / np.var(firing_rate_list_centered))

    return spatial_comparison_t


def temporal_variance_spatial_change_detection(serial_idxes, epoch='interval'):

    gaussian_center1 = np.arange(6., 32.-6, 2)
    gaussian_center2 = np.array([15.]*len(gaussian_center1))
    signal2_strength = 1.

    batch_size = len(gaussian_center1)

    spatial_change_detection_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory + '/model/' + 'spatial_change_detection/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_change_detection', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2, signal2_strength=signal2_strength)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        spatial_change_detection_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return spatial_change_detection_t


def temporal_variance_ctx_decision_making(serial_idxes, epoch='interval', noise_on=False, mode='pilot'):

    c = np.array([0.04, 0.02, 0.01, -0.01, -0.02, -0.04])
    batch_size = len(c)
    gamma_bar = np.array([1.] * batch_size).flatten()
    choice = np.array([1.] * batch_size).flatten()
    signal2_strength = 1.

    decision_making_no_timing_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'ctx_decision_making/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='ctx_decision_making', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar, c=c, choice=choice, signal2_strength=signal2_strength)

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        ##########################################
        # shape: (time, c, neuron)
        firing_rate = np.concatenate(list(firing_rate_binder[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)), axis=1)
        # shape: neuron, time, c,
        firing_rate_list = firing_rate.transpose((2, 0, 1))

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        decision_making_no_timing_t.append(np.var(firing_rate_list_t) / np.var(firing_rate_list_centered))

    return decision_making_no_timing_t


def feedback_current(serial_idxes, noise_on=False):
    gaussian_center1 = np.arange(6., 32.-6, 2)
    signal2_strength = 1.
    batch_size = len(gaussian_center1)
    # sky blue, green, pink, kelly-vivid-yellow
    color_list = ['#75bbfd', '#15b01a', '#ff81c0', '#FFB300']

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'spatial_reproduction/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center=gaussian_center1, signal2_strength=signal2_strength)

        feedback_binder = run_result.feedback_current_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs['interval']

        feedback = np.concatenate(
            list(feedback_binder[stim1_off[i]:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        print(feedback.shape)
        # shape: time, batch_for_direction, neuron
        feedback_mean = np.mean(feedback, axis=(1))

        fig = plt.figure(figsize=(2.5, 2.1))
        ax = fig.add_axes([0.25, 0.2, 0.63, 0.65])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.plot(np.arange(len(feedback_mean)) * 20, feedback_mean, color=color_list[0])

        for i in range(feedback.shape[1]):
            plt.plot(np.arange(len(feedback[:, i])) * 20, feedback[:, i], color='black', alpha=0.2)

        ax.set_xlabel('Time (ms)', fontsize=fs)
        ax.set_ylabel('Feedback current (a.u.)', fontsize=fs)

        plt.show()
        return fig

