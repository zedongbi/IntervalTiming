from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

import os

from .. import run

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

root_directory = os.path.join(os.getcwd(),'core')

transient_period = 20


def temporal_and_stim_variance_spatial_reproduction(serial_idxes, noise_on=False):

    spatial_reproduction_t = list()
    spatial_reproduction_s = list()

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

        firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)
        spatial_reproduction_s.append(np.var(firing_rate_list_s)/np.var(firing_rate_list_centered))

    return spatial_reproduction_t, spatial_reproduction_s


def temporal_and_stim_variance_timed_spatial_reproduction(serial_idxes, noise_on=False):

    timed_spatial_reproduction_t = list()
    timed_spatial_reproduction_s = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory+'/model/'+'timed_spatial_reproduction/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32.-6., 2.)
        batch_size = len(ring_centers)
        prod_intervals = np.array([1200] * batch_size)
        dly_intervals = np.array([1600] * batch_size)

        #print(model_dir)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gaussian_center=ring_centers)

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
        timed_spatial_reproduction_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))
        firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)
        timed_spatial_reproduction_s.append(np.var(firing_rate_list_s)/np.var(firing_rate_list_centered))

    return timed_spatial_reproduction_t, timed_spatial_reproduction_s


def temporal_variance(serial_idxes, noise_on=False):

    ###############################################################################################
    timed_spatial_reproduction_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory+'/model/'+'timed_spatial_reproduction/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32.-6., 2.)
        batch_size = len(ring_centers)
        prod_intervals = np.array([1200] * batch_size)
        dly_intervals = np.array([1600] * batch_size)

        #print(model_dir)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gaussian_center=ring_centers)

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
        timed_spatial_reproduction_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    ###############################################################################################
    timed_spatial_reproduction_broad_tuning_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory+'/model/'+'timed_spatial_reproduction_broad_tuning/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(12., 32.+12-12., 2.)
        batch_size = len(ring_centers)
        prod_intervals = np.array([1200] * batch_size)
        dly_intervals = np.array([1600] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction_broad_tuning', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gaussian_center=ring_centers)

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
        timed_spatial_reproduction_broad_tuning_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    ###############################################################################################
    spatial_reproduction_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory+'/model/'+'spatial_reproduction/'+str(serial_idx)

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

    ###############################################################################################
    spatial_reproduction_broad_tuning_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory+'/model/'+'spatial_reproduction_broad_tuning/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(12., 32.+12-12., 2.)
        batch_size = len(ring_centers)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_reproduction_broad_tuning', is_cuda=False, noise_on=noise_on)
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
        spatial_reproduction_broad_tuning_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    ###############################################################################################
    spatial_comparison_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_comparison/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)
        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_comparison', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        spatial_comparison_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    ###############################################################################################
    spatial_comparison_broad_tuning_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_comparison_broad_tuning/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(12., 32.+12-12., 2.)
        gaussian_center2 = np.array([22.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_comparison_broad_tuning', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

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
        spatial_comparison_broad_tuning_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    ###############################################################################################
    spatial_change_detection_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_change_detection/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_change_detection', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

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

    ###############################################################################################
    spatial_change_detection_broad_tuning_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_change_detection_broad_tuning/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(12., 32.+12-12., 2.)
        gaussian_center2 = np.array([22.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_change_detection_broad_tuning', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

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
        spatial_change_detection_broad_tuning_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return timed_spatial_reproduction_t, timed_spatial_reproduction_broad_tuning_t, \
           spatial_reproduction_t, spatial_reproduction_broad_tuning_t,\
           spatial_comparison_t, spatial_comparison_broad_tuning_t,\
           spatial_change_detection_t, spatial_change_detection_broad_tuning_t


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


def temporal_variance_spatial_change_detection(serial_idxes, noise_on=False):

    spatial_change_detection_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_change_detection/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_change_detection', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'interval'
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


def temporal_variance_spatial_comparison(serial_idxes, noise_on=False):

    spatial_comparison_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_comparison/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_comparison', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        spatial_comparison_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return spatial_comparison_t


def temporal_variance_spatial_reproduction_variable_delay(serial_idxes, noise_on=False):

    frequency_no_timing_t = list()

    for serial_idx in serial_idxes:

        model_dir = root_directory + '/model/'+'spatial_reproduction_variable_delay/'+str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32.-6., 2.)
        batch_size = len(ring_centers)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_reproduction_variable_delay', is_cuda=False, noise_on=noise_on)
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
        frequency_no_timing_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return frequency_no_timing_t


def temporal_variance_spatial_comparison_variable_delay(serial_idxes, noise_on=False):

    frequency_comparison_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_comparison_variable_delay/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_comparison_variable_delay', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        frequency_comparison_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return frequency_comparison_t


def temporal_variance_spatial_change_detection_variable_delay(serial_idxes, noise_on=False):

    frequency_change_detection_t = list()

    for serial_idx in serial_idxes:
        model_dir = root_directory+'/model/' + 'spatial_change_detection_variable_delay/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        gaussian_center1 = np.arange(6., 32.-6., 2.)
        gaussian_center2 = np.array([15.] * len(gaussian_center1))
        batch_size = len(gaussian_center1)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='spatial_change_detection_variable_delay', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, gaussian_center1=gaussian_center1, gaussian_center2=gaussian_center2)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'interval'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = np.concatenate(
            list(firing_rate[stim1_off[i]+transient_period:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
            axis=1).squeeze()

        # shape: neuron, time, batch_for_direction,
        firing_rate_list = firing_rate_list.transpose((2, 0, 1))
        firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean
        firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
        frequency_change_detection_t.append(np.var(firing_rate_list_t)/np.var(firing_rate_list_centered))

    return frequency_change_detection_t
