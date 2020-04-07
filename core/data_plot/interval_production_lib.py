from __future__ import division

import matplotlib as mpl

import matplotlib.patches as patches


import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

from .. import run
from .analysis import PCASpace
from .analysis import ScalingSpace

plt.style.use('default')
fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"


def neuron_activity_plot_three_epochs(serial_idx=0, dly_interval=1600):
    _color_list = ['#FFB300', '#803E75', '#FF6800', '#A6BDD7', '#C10020', '#CEA262', '#817066', '#007D34', '#F6768E', '#00538A', '#FF7A5C', '#53377A', '#FF8E00', '#B32851', '#F4C800', '#7F180D', '#93AA00', '#593315', '#F13A13', '#232C16']

    model_dir = './core/model/'+'interval_production/'+str(serial_idx)

    prod_intervals = np.array([900])#np.array([600, 1200])

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
    neuron_number = int(runnerObj.model.hidden_size)

    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    _, end_time = trial_input.epochs['go']
    start_time = np.zeros_like(end_time)

    stim1_on, stim1_off = trial_input.epochs['stim1']
    stim2_on, stim2_off = trial_input.epochs['stim2']
    control_on, control_off = trial_input.epochs['go_cue']

    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.6])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.tick_params(axis="x", labelsize=fs)
    ax.tick_params(axis="y", labelsize=fs)

    ax.set_ylabel('Firing rate (a.u.)', fontsize=fs)
    ax.set_xlabel('Time (ms)', fontsize=fs)

    batch_idx = 0
    firing_rate = run_result.firing_rate_binder[start_time[batch_idx]:end_time[batch_idx], batch_idx, :].detach().cpu().numpy()
    max_rate_idx = np.argsort(np.mean(firing_rate, axis=0))[::-1][0:10]
    firing_rate = firing_rate[:, max_rate_idx]

    for neuron_idx in range(0, firing_rate.shape[1]):
        plt.plot(np.array(range(0, firing_rate.shape[0])) * 20, firing_rate[:, neuron_idx], color=_color_list[neuron_idx])

    ymax = ax.get_ylim()[1]
    ax.add_patch(patches.Rectangle((stim1_on[batch_idx] * 20, 0), (stim1_off[batch_idx]-stim1_on[batch_idx]) * 20, ymax, fc='blue', alpha=0.5))
    ax.add_patch(patches.Rectangle((stim2_on[batch_idx] * 20, 0), (stim2_off[batch_idx]-stim2_on[batch_idx]) * 20, ymax, fc='blue', alpha=0.5))
    ax.add_patch(patches.Rectangle((control_on[batch_idx] * 20, 0), (control_off[batch_idx]-control_on[batch_idx]) * 20, ymax, fc='blue', alpha=0.5))
    ax.text(1200, 31, '$T$=900ms', fontsize=fs)
    ax.text(100, 39, 'Perc.\nepoch'.center(14), fontsize=fs)
    ax.text(1300, 39, 'Delay\nepoch'.center(12), fontsize=fs)
    ax.text(2700, 39, 'Prod.\nepoch'.center(14), fontsize=fs)

    plt.show()
    return fig


def PCA_plot(serial_idx=0, epoch='interval', dly_interval=1600, prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), noise_on=False):
    model_dir = './core/model/'+'interval_production/'+str(serial_idx)

    batch_size = len(prod_intervals)
    alphas = np.linspace(0, 1, batch_size)
    _color_list = list(map(cm.rainbow, alphas))# rainbow color

    dly_intervals = np.array([dly_interval] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

    stim1_off, stim2_on = trial_input.epochs[epoch]
    if epoch == 'go':
        outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
        motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
        stim2_on = stim1_off + motion_time

    firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))

    concate_firing_rate = np.concatenate(firing_rate_list, axis=0)

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)
    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    time_size = stim2_on - stim1_off

    delim = np.cumsum(time_size)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)

    fig = plt.figure(figsize=(3, 3))

    ax = fig.gca(projection='3d')
    for i in range(0, len(concate_transform_split)):

        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], color=_color_list[i])
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  marker='*', color=_color_list[i])
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  marker='o', color=_color_list[i])

    ax.set_xlabel('PC1', fontsize=fs,labelpad=-5)
    ax.set_ylabel('PC2', fontsize=fs,labelpad=-5)
    ax.set_zlabel('PC3', fontsize=fs,labelpad=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # color bar
    fig2 = plt.figure(figsize=(3, 2.1))
    ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    cmap = plt.get_cmap('rainbow', 7)
    mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    ax2.set_yticklabels(['600', '900', '1200'], size=fs)
    ax2.set_title('Time interval\n(ms)', fontsize=fs)

    plt.show()
    return fig, fig2


def neuron_activity_example_plot(serial_idx=0, epoch='interval', dly_interval=1600, prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), noise_on=False):
    plot_neuron_number = 8
    model_dir = './core/model/'+'interval_production/'+str(serial_idx)

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()
    stim1_off, stim2_on = trial_input.epochs[epoch]

    # get the time point when the output becomes larger than 0.5, and regarding this time point as the time of the movement
    if epoch == 'go':
        outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
        motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
        stim2_on = stim1_off + motion_time

    firing_rate_list = [firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :] for batch_idx in range(0, batch_size)]

    # choose the neurons with largest fluctuations to plot
    fluc_amp = [(np.max(firing_rate_list[batch_idx], axis=0)-np.min(firing_rate_list[batch_idx], axis=0))[np.newaxis, :] for batch_idx in range(0, batch_size)]
    fluc_amp = np.concatenate(fluc_amp, axis=0)
    max_fluc_amp = np.max(fluc_amp, axis=0)
    max_fluc_neuron_idx = np.argsort(max_fluc_amp)[-plot_neuron_number:]
    choice_idx = np.array([2, 6])
    max_fluc_neuron_idx = max_fluc_neuron_idx[choice_idx]
    firing_rate_list = [x[:, max_fluc_neuron_idx] for x in firing_rate_list]

    alphas = np.linspace(0, 1, batch_size)
    _color_list = list(map(cm.rainbow, alphas))#sns.color_palette("hls", batch_size)

    f, ((ax0, ax1)) = plt.subplots(2, 1, figsize=(1.25, 2.1))

    for prod_interval_idx in reversed(range(batch_size)):
        data = firing_rate_list[prod_interval_idx][:, 0]
        ax0.plot(np.array(range(0, len(data))) * 20, data, color=_color_list[prod_interval_idx])
        data = firing_rate_list[prod_interval_idx][:, 1]
        ax1.plot(np.array(range(0, len(data))) * 20, data, color=_color_list[prod_interval_idx])

    ax0.axis('off')
    ax1.axis('off')

    ax1.set_ylim(bottom=-3)
    ax1.set_xlim(left=-70)

    ax0.set_xlim(left=-70)
    ax1.set_xlim(left=-70)

    ax1.plot([0, 500], [-2, -2], color='black')
    ax1.plot([-50, -50], [0, 10], color='black')

    plt.show()
    return f


def explained_var_interval_r2(serial_idxes, epoch='interval', dly_interval=1600, prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), noise_on=False):

    result = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        batch_size = len(prod_intervals)
        dly_intervals = np.array([dly_interval] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()
        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_list = [firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :] for batch_idx in range(0, batch_size)]

        total_var = [np.sum(np.square(x - np.mean(x))) for x in firing_rate_list[0:-1]]
        unexplained_var = [np.sum(np.square(x - firing_rate_list[-1][0:x.shape[0], :])) for x in firing_rate_list[0:-1]]

        for x, y in zip(total_var, unexplained_var):
            result.append(1 - y / x)

    result = np.array(result)
    mean = np.mean(result)
    std = np.std(result)

    fig = plt.figure(figsize=(2.5, 1))
    ax = fig.add_axes([0.3, 0.15, 0.8/1.5/4, 0.73])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    plt.bar([0], [mean], yerr=[std], alpha=0.5)
    plt.xticks([0], [''], fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.gca().set_ylim([0.9, 1])
    plt.ylabel(r'$R^2$', fontsize=fs)

    plt.show()
    return fig


def PCA_velocity_plot(serial_idx=0, epoch='delay', prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), dly_interval=1600):
    model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    _color_list = np.arange(1, batch_size+1)/(batch_size)#sns.color_palette("hls", batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

    stim1_off, stim2_on = trial_input.epochs[epoch]
    concate_firing_rate = np.concatenate([firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :] for batch_idx in range(0, batch_size)], axis=0)

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)
    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    time_size = stim2_on - stim1_off
    accum_time_size = np.cumsum(time_size)
    concate_transform_split = np.split(concate_firing_rate_transform, accum_time_size[:-1], axis=0)
    concate_original_split = np.split(concate_firing_rate, accum_time_size[:-1], axis=0)

    velocity = [np.sqrt(np.sum(np.diff(x, axis=0) ** 2, axis=1)) for x in concate_original_split]


    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.gca(projection='3d')

    ax.set_xlabel('PC1', fontsize=fs)
    ax.set_ylabel('PC2', fontsize=fs)
    ax.set_zlabel('PC3', fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    xmin = min([np.min(x[:, 0]) for x in concate_transform_split])
    xmax = max([np.max(x[:, 0]) for x in concate_transform_split])

    ymin = min([np.min(x[:, 1]) for x in concate_transform_split])
    ymax = max([np.max(x[:, 1]) for x in concate_transform_split])

    zmin = min([np.min(x[:, 2]) for x in concate_transform_split])
    zmax = max([np.max(x[:, 2]) for x in concate_transform_split])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    max_velocity = max([np.max(x) for x in velocity])
    min_velocity = min([np.min(x) for x in velocity])
    norm = plt.Normalize(min_velocity, max_velocity)

    lc = [Line3DCollection(np.concatenate([x.reshape(-1, 1, 3)[:-1], x.reshape(-1, 1, 3)[1:]], axis=1), cmap='rainbow', norm=norm) for x in concate_transform_split]

    for i in range(len(lc)):
        lc[i].set_array(velocity[i])
        ax.add_collection3d(lc[i])

    m = cm.ScalarMappable(cmap='rainbow')
    m.set_array([zmin, zmax])
    m.set_clim(vmin=min_velocity, vmax=max_velocity)

    cbar = fig.colorbar(m, shrink=0.8, pad=0.2)
    #fig.colorbar(m, fraction=0.046, pad=0.04)
    cbar.ax.set_title('Velocity (a.u.)', fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)

    for batch_idx in range(batch_size):

        ax.scatter(concate_transform_split[batch_idx][-1, 0], concate_transform_split[batch_idx][-1, 1], concate_transform_split[batch_idx][-1, 2], marker='o', color='black', alpha=_color_list[batch_idx])
        ax.scatter(concate_transform_split[batch_idx][0, 0], concate_transform_split[batch_idx][0, 1], concate_transform_split[batch_idx][0, 2], marker='*', color='black', alpha=_color_list[batch_idx])

    for batch_idx in range(batch_size-1):
        ax.plot([concate_transform_split[batch_idx][-1, 0], concate_transform_split[batch_idx+1][-1, 0]],
                [concate_transform_split[batch_idx][-1, 1], concate_transform_split[batch_idx+1][-1, 1]],
                [concate_transform_split[batch_idx][-1, 2], concate_transform_split[batch_idx+1][-1, 2]], '--', color='black')

    plt.show()
    return fig


def velocity_during_delay_plot_batch(serial_idxes, epoch='delay', prod_intervals=np.array([600, 1200]), dly_interval=1600):

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    _alpha_list = np.arange(1, batch_size + 1) / (batch_size)  # sns.color_palette("hls", batch_size)

    velocity_list = list()

    for serial_idx in serial_idxes:
        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]
        firing_rate_list = [firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :] for batch_idx in range(0, batch_size)]
        velocity = np.concatenate([np.sqrt(np.sum(np.diff(x, axis=0) ** 2, axis=1))[np.newaxis, :] for x in firing_rate_list], axis=0)

        velocity_list.append(velocity[np.newaxis, :, :])

    velocity_list = np.concatenate(velocity_list, axis=0)
    velocity_mean = np.mean(velocity_list, axis=0)
    velocity_sem = np.std(velocity_list, axis=0)/np.sqrt(velocity_list.shape[0])


    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.4, 0.25, 0.5, 0.7])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    for i in range(1, velocity_mean.shape[0]-1):
        plt.plot(np.arange(len(velocity_mean[i, :])) * 20, velocity_mean[i, :], color='black', alpha=_alpha_list[i])
    i = 0
    plt.plot(np.arange(len(velocity_mean[i, :])) * 20, velocity_mean[i, :], color='blue')

    plt.fill_between(np.arange(len(velocity_mean[i,:]))*20, velocity_mean[i,:]-velocity_sem[i,:], velocity_mean[i,:]+velocity_sem[i,:], color='blue', alpha=0.2)

    i = velocity_mean.shape[0]-1
    plt.plot(np.arange(len(velocity_mean[i, :])) * 20, velocity_mean[i, :], color='red')

    plt.fill_between(np.arange(len(velocity_mean[i,:]))*20, velocity_mean[i,:]-velocity_sem[i,:], velocity_mean[i,:]+velocity_sem[i,:], color='red', alpha=0.2)

    plt.gca().set_xlabel('Time (ms)', fontsize=fs)
    plt.gca().set_ylabel('Velocity (a.u.)', fontsize=fs)

    plt.xticks([0, 800, 1600], fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.show()
    return fig


def end_delay_dim_plot_batch(serial_idxes, epoch='delay', prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), dly_interval=1600, pca_n=5):

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    _alpha_list = np.arange(1, batch_size + 1) / (batch_size)  # sns.color_palette("hls", batch_size)

    explained_variance_ratio_list = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)
        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]
        firing_rate_list = [firing_rate_binder[stim2_on[batch_idx]-1, batch_idx, :][np.newaxis, :] for batch_idx in range(0, batch_size)]

        concate_firing_rate = np.concatenate(firing_rate_list, axis=0)

        pca = PCA(n_components=pca_n)
        pca.fit(concate_firing_rate)

        explained_variance_ratio_list.append(pca.explained_variance_ratio_[np.newaxis, :])
    explained_variance_ratio_list = np.concatenate(explained_variance_ratio_list, axis=0)

    explained_variance_ratio_mean = np.mean(explained_variance_ratio_list, axis=0)
    explained_variance_ratio_sem = np.std(explained_variance_ratio_list, axis=0)/np.sqrt(explained_variance_ratio_list.shape[0])

    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.2, 0.2, 0.4, 0.8])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.plot(np.arange(1, 1+len(explained_variance_ratio_mean)), explained_variance_ratio_mean, color='black', marker='o')
    plt.gca().set_xlabel('PC', fontsize=fs)
    plt.gca().set_ylabel('Explained Var. Ratio', fontsize=fs)

    plt.xticks([1,3,5], fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.show()
    return fig


def end_delay_topology(serial_idxes, epoch='delay', prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), dly_interval=1600, pca_n=5):

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    _alpha_list = np.arange(1, batch_size + 1) / (batch_size)  # sns.color_palette("hls", batch_size)

    result_list = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]
        firing_rate_list = [firing_rate_binder[stim2_on[batch_idx], batch_idx, :][np.newaxis, :] for batch_idx in range(0, batch_size)]

        concate_firing_rate = np.concatenate(firing_rate_list, axis=0)

        pca = PCA(n_components=1)
        pca.fit(concate_firing_rate)
        firing_rate_transform = pca.transform(concate_firing_rate)
        min = np.min(firing_rate_transform)
        max = np.max(firing_rate_transform)
        firing_rate_transform = (firing_rate_transform - min)/(max-min)
        if firing_rate_transform[0, 0] > firing_rate_transform[-1, 0]:
            firing_rate_transform = firing_rate_transform[::-1, :]
        result_list.append(firing_rate_transform)

    result_list = np.concatenate(result_list, axis=1)

    result_mean = np.mean(result_list, axis=1)

    #print([x.shape for x in velocity])
    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.25, 0.22, 0.3, 0.75])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    for i in range(result_list.shape[1]):
        plt.plot(prod_intervals, result_list[:, i], color='black', alpha=0.2)
    plt.plot(prod_intervals, result_mean, color='blue')

    plt.gca().set_xlabel('Time Interval $T$ (ms)', fontsize=fs)
    plt.gca().set_ylabel('Relative position', fontsize=fs)

    plt.xticks([600, 900, 1200], fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.show()
    return fig


def adjacent_trajectory_distance_delay_epoch(serial_idxes, prod_intervals=np.arange(600, 1220, 20), dly_interval=1600):
    epoch = 'delay'

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    _alpha_list = np.arange(1, batch_size + 1) / (batch_size)  # sns.color_palette("hls", batch_size)

    distance_list = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        ###############################################
        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]
        stim1_off = stim1_off + 0
        #shape: (interval, time, neuron)
        firing_rate = np.concatenate(
            [firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :][np.newaxis, :, :] for batch_idx in
             range(0, batch_size)], axis=0)

        firing_rate1 = firing_rate[0:-1, :, :]
        firing_rate2 = firing_rate[1:, :, :]
        adj_distance = np.sqrt(np.sum((firing_rate1 - firing_rate2)**2, axis=2))
        adj_distance = adj_distance/adj_distance[:, [0]]
        distance_list.append(adj_distance[np.newaxis, :, :])

    distance_list = np.concatenate(distance_list, axis=0)
    distance_list = distance_list.reshape(-1, distance_list.shape[-1])
    distance_list_mean = np.mean(distance_list, axis=0)
    distance_list_std = np.std(distance_list, axis=0)

    fig1 = plt.figure(figsize=(2.1, 2.1))
    #ax = fig1.add_axes([0.25, 0.21, 0.35, 0.7])
    ax = fig1.add_axes([0.25, 0.21, 0.65, 0.7])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(np.arange(len(distance_list_mean))*20, distance_list_mean)
    plt.fill_between(np.arange(len(distance_list_mean))*20, distance_list_mean-distance_list_std, distance_list_mean+distance_list_std, color='blue', alpha=0.2)

    plt.xlabel('Time (ms)', fontsize=fs)
    plt.ylabel('Normalized Distance', fontsize=fs)

    plt.xticks([0, 800, 1600], fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()

    return fig1


def neuron_tuning_end_of_delay_grid_plot(serial_idx=0, func_relevance_threshold=2):
    model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

    prod_intervals = np.arange(600, 1220, 20)
    dly_interval = 1600

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

    start_delay, end_delay = trial_input.epochs['delay']

    firing_rate = np.concatenate([firing_rate_binder[end_delay[batch_idx], batch_idx, :][np.newaxis, :] for batch_idx in range(0, batch_size)], axis=0)
    func_relevance_neuron = np.argwhere(np.sum(firing_rate>func_relevance_threshold, axis=0) > 0).squeeze()
    firing_rate = firing_rate[:, func_relevance_neuron]

    separator1 = int((batch_size-1)/4)
    separator2 = int((batch_size-1)/2)
    separator3 = int((batch_size-1)/4*3)

    #print(separator1, separator2, separator3)
    #print(firing_rate.shape)

    mean_rate_1 = np.mean(firing_rate[0:separator1, :], axis=0)
    mean_rate_2 = np.mean(firing_rate[separator1:separator2, :], axis=0)
    mean_rate_3 = np.mean(firing_rate[separator2:separator3, :], axis=0)
    mean_rate_4 = np.mean(firing_rate[separator3:, :], axis=0)

    mono_increase = np.argwhere(((mean_rate_1<=mean_rate_2)*(mean_rate_2<=mean_rate_3)*(mean_rate_3<=mean_rate_4)) > 0).flatten()
    mono_decrease = np.argwhere(((mean_rate_1>=mean_rate_2)*(mean_rate_2>=mean_rate_3)*(mean_rate_3>=mean_rate_4)) > 0).flatten()
    other_neuron = np.array(list(set(np.arange(firing_rate.shape[1])) - set(mono_increase.flatten()) - set(mono_decrease.flatten())))

    firing_rate = np.concatenate([firing_rate[:, mono_increase], firing_rate[:, mono_decrease], firing_rate[:, other_neuron]], axis=1)

    # normalize
    for i in range(0, firing_rate.shape[1]):
        firing_rate[:, i] = (firing_rate[:, i]-np.min(firing_rate[:, i])) / (np.max(firing_rate[:, i])-np.min(firing_rate[:, i]))

    X, Y = np.mgrid[0:firing_rate.shape[0]+1, 0:firing_rate.shape[1]+1]

    fig = plt.figure(figsize=(2.6, 2.4))
    ax = fig.add_axes([0.3, 0.2, 0.63, 0.62])

    plt.gca().set_xlabel('Time interval $T$ (ms)', fontsize=fs)
    plt.gca().set_ylabel('Neuron', fontsize=fs)

    plt.xticks([0.5, (len(prod_intervals)-1)/3+0.5, (len(prod_intervals)-1)/3*2+0.5, len(prod_intervals)-0.5], [600, 800, 1000, 1200], fontsize=fs)
    plt.yticks([len(mono_increase)/2, len(mono_increase)+len(mono_decrease)/2, len(mono_increase)+len(mono_decrease)+len(other_neuron)/2],
               ['MoI', 'MoD', 'non-M'], fontsize=fs)

    # Make the plot
    cmap = mpl.rcParams["image.cmap"] # plt.get_cmap('rainbow')
    plt.pcolormesh(X, Y, firing_rate, cmap=cmap)
    plt.plot([0, len(prod_intervals)],[len(mono_increase), len(mono_increase)], '--', color='black')
    plt.plot([0, len(prod_intervals)],[len(mono_increase)+len(mono_decrease), len(mono_increase)+len(mono_decrease)], '--',  color='black')

    # color bar
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)

    cbar = fig.colorbar(m)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.set_title('Normalized\n activity', fontsize=fs)
    plt.show()
    return fig


def neuron_type_end_of_delay(serial_idxes, func_relevance_threshold=2):
    prod_intervals = np.arange(600, 1220, 20)
    dly_interval = 1600

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    moi_portion = list()
    mod_portion = list()
    other_portion = list()

    for serial_idx in serial_idxes:
        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        start_delay, end_delay = trial_input.epochs['delay']

        firing_rate = np.concatenate([firing_rate_binder[end_delay[batch_idx], batch_idx, :][np.newaxis, :] for batch_idx in range(0, batch_size)], axis=0)
        func_relevance_neuron = np.argwhere(np.sum(firing_rate>func_relevance_threshold, axis=0) > 0).squeeze()
        firing_rate = firing_rate[:, func_relevance_neuron]

        separator1 = int((batch_size - 1) / 4)
        separator2 = int((batch_size - 1) / 2)
        separator3 = int((batch_size - 1) / 4 * 3)

        # print(separator1, separator2, separator3)
        # print(firing_rate.shape)

        mean_rate_1 = np.mean(firing_rate[0:separator1, :], axis=0)
        mean_rate_2 = np.mean(firing_rate[separator1:separator2, :], axis=0)
        mean_rate_3 = np.mean(firing_rate[separator2:separator3, :], axis=0)
        mean_rate_4 = np.mean(firing_rate[separator3:, :], axis=0)

        mono_increase = np.argwhere(
            ((mean_rate_1 <= mean_rate_2) * (mean_rate_2 <= mean_rate_3) * (mean_rate_3 <= mean_rate_4)) > 0).flatten()
        mono_decrease = np.argwhere(
            ((mean_rate_1 >= mean_rate_2) * (mean_rate_2 >= mean_rate_3) * (mean_rate_3 >= mean_rate_4)) > 0).flatten()
        other_neuron = np.array(
            list(set(np.arange(firing_rate.shape[1])) - set(mono_increase.flatten()) - set(mono_decrease.flatten())))

        moi_portion.append(len(mono_increase)/len(func_relevance_neuron))
        mod_portion.append(len(mono_decrease)/len(func_relevance_neuron))
        other_portion.append(len(other_neuron)/len(func_relevance_neuron))

    moi_portion = np.array(moi_portion)
    mod_portion = np.array(mod_portion)
    other_portion = np.array(other_portion)

    low_tune_neuron_mean = np.mean(moi_portion)
    high_tune_neuron_mean = np.mean(mod_portion)
    middle_tune_neuron_mean = np.mean(other_portion)

    low_tune_neuron_sem = np.std(moi_portion)/np.sqrt(len(moi_portion))
    high_tune_neuron_sem = np.std(mod_portion)/np.sqrt(len(mod_portion))
    middle_tune_neuron_sem = np.std(other_portion)/np.sqrt(len(other_portion))

    fig = plt.figure(figsize=(2.5, 2.4))
    #ax = fig.add_axes([0.2, 0.15, 0.8/1.5*0.8, 0.7])
    ax = fig.add_axes([0.2, 0.15, 0.8/1.5*0.9, 0.7])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.bar([0, 1, 2], [low_tune_neuron_mean, high_tune_neuron_mean, middle_tune_neuron_mean], yerr=[low_tune_neuron_sem, high_tune_neuron_sem, middle_tune_neuron_sem], alpha=0.5)
    plt.xticks([0, 1, 2], ['MoI', 'MoD', 'non-M'], fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.ylabel('Portion', fontsize=fs)

    #ax.annotate('Total='+f'{(low_tune_neuron_mean+high_tune_neuron_mean):.2f}', xy=(0.333, 1.0), xytext=(0.333, 1.10), xycoords='axes fraction',
    #            fontsize=fs, ha='center', va='bottom',
    #            arrowprops=dict(arrowstyle='-[, widthB=2.67, lengthB=0.5', lw=1.0))
    ax.annotate('Total='+f'{(low_tune_neuron_mean+high_tune_neuron_mean):.2f}', xy=(0.333, 1.0), xytext=(0.333, 1.10), xycoords='axes fraction',
                fontsize=fs, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-[, widthB=2., lengthB=0.5', lw=1.0))

    plt.show()
    return fig


def neuron_activity_example_plot_no_scaling_vs_scaling(serial_idx=0, epoch='interval', dly_interval=1600, prod_intervals=np.array([600, 700, 800, 900, 1000, 1100, 1200]), noise_on=False):
    plot_neuron_number = 8
    model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

    batch_size = len(prod_intervals)
    dly_intervals = np.array([dly_interval] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()
    stim1_off, stim2_on = trial_input.epochs[epoch]

    if epoch == 'go':
        outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
        motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
        stim2_on = stim1_off + motion_time

    firing_rate_list = [firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :] for batch_idx in range(0, batch_size)]

    fluc_amp = [(np.max(firing_rate_list[batch_idx], axis=0)-np.min(firing_rate_list[batch_idx], axis=0))[np.newaxis, :] for batch_idx in range(0, batch_size)]
    #fluc_amp = [(np.max(firing_rate_list[batch_idx], axis=0))[np.newaxis, :] for batch_idx in range(0, batch_size)]
    fluc_amp = np.concatenate(fluc_amp, axis=0)
    max_fluc_amp = np.max(fluc_amp, axis=0)
    max_fluc_neuron_idx = np.argsort(max_fluc_amp)[-plot_neuron_number:]
    choice_idx = np.array([0, 1])
    max_fluc_neuron_idx = max_fluc_neuron_idx[choice_idx]
    firing_rate_list = [x[:, max_fluc_neuron_idx] for x in firing_rate_list]

    alphas = np.linspace(0, 1, batch_size)
    _color_list = list(map(cm.rainbow, alphas))#sns.color_palette("hls", batch_size)

    #plt.figure(figsize=(2.5, 2.1))

    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(2.5, 2.1))

    for prod_interval_idx in range(batch_size):
        data = firing_rate_list[prod_interval_idx][:, 0]
        ax0.plot(np.array(range(0, len(data))) * 20, data, color=_color_list[prod_interval_idx])
        data = firing_rate_list[prod_interval_idx][:, 1]
        ax1.plot(np.array(range(0, len(data))) * 20, data, color=_color_list[prod_interval_idx])

    for prod_interval_idx in range(batch_size):
        data = firing_rate_list[prod_interval_idx][:, 0]
        ax2.plot(np.array(range(0, len(data))) * 20 / motion_time[prod_interval_idx], data, color=_color_list[prod_interval_idx])
        data = firing_rate_list[prod_interval_idx][:, 1]
        ax3.plot(np.array(range(0, len(data))) * 20 / motion_time[prod_interval_idx], data, color=_color_list[prod_interval_idx])

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax0.set_ylim(bottom=-3)
    ax0.set_xlim(left=-70)
    ax0.plot([0, 500], [-2, -2], color='black')
    ax0.plot([-50, -50], [0, 10], color='black')

    ax0.set_xlim(left=-70)
    ax1.set_xlim(left=-70)

    plt.show()
    return f


def production_epoch_explained_variance_plot_batch(serial_idxes, epoch='go', dim_n=9, dly_interval=1600):

    scaling_idx_list = list()
    explained_variance_ratio_cumsum_list = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        print(serial_idx)

        if not os.path.exists(model_dir):
            continue

        prod_intervals = np.array([600, 700, 800, 900, 1000, 1100, 1200])

        batch_size = len(prod_intervals)
        dly_intervals = np.array([dly_interval] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]

        if epoch == 'go':
            outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
            motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
            stim2_on = stim1_off + motion_time

        firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))

        pca_obj = PCASpace.PCASpace(firing_rate_list, dim_n, model_dir)

        total_explained_variance_ratio = np.sum(pca_obj.pca.explained_variance_ratio_)

        pca_space_data = pca_obj.result  # tools.load_pickle(os.path.join(model_dir, 'pca_space' + '.pkl'))
        data_list = pca_space_data['data_pca']

        scaling_dir_name = os.path.join(model_dir, 'scaling_space' + '.pkl')

        ScalingSpaceObj = ScalingSpace.ScalingSpace(data_list, scaling_dir_name)

        scaling_dir_name = os.path.join(model_dir, 'scaling_space' + '.pkl')

        scaling_space_data = ScalingSpaceObj.result  # tools.load_pickle(scaling_dir_name)

        data_rescale = scaling_space_data['data_rescale']
        scaling_transform_matrix = scaling_space_data['scaling_transform_matrix']

        parameter_n, ref_time_n, pca_n = data_rescale.shape

        variance_explained_ratio = ScalingSpace.variance_explained(np.reshape(data_rescale, (-1, pca_n)),
                                                                   scaling_transform_matrix)

        scaling_idx = np.zeros((dim_n,))
        for pca_idx in range(0, dim_n):
            scaling_idx[pca_idx] = ScalingSpace.scaling_index_multi_dim(data_rescale, scaling_transform_matrix[:, 0:(pca_idx+1)])

        explained_variance_ratio_cumsum = np.cumsum(variance_explained_ratio)

        scaling_idx_list.append(scaling_idx[np.newaxis, :])
        explained_variance_ratio_cumsum_list.append(explained_variance_ratio_cumsum[np.newaxis, :] * total_explained_variance_ratio)

    scaling_idx_list = np.concatenate(scaling_idx_list, axis=0)
    explained_variance_ratio_cumsum_list = np.concatenate(explained_variance_ratio_cumsum_list, axis=0)


    scaling_idx_mean = np.mean(scaling_idx_list, axis=0)
    scaling_idx_sem = np.std(scaling_idx_list, axis=0)/np.sqrt(scaling_idx_list.shape[0])
    explained_variance_ratio_cumsum_mean = np.mean(explained_variance_ratio_cumsum_list, axis=0)
    explained_variance_ratio_cumsum_sem = np.std(explained_variance_ratio_cumsum_list, axis=0)/np.sqrt(explained_variance_ratio_cumsum_list.shape[0])


    fig = plt.figure(figsize=(2.3, 2.1))
    #ax = fig.add_axes([0.2, 0.2, 0.75, 0.65])
    ax = fig.add_axes([0.2, 0.2, 0.75*0.66, 0.65*0.85])

    ax.errorbar(np.arange(1, 1+dim_n), scaling_idx_mean, yerr=scaling_idx_sem, label='scaling index')
    ax.errorbar(np.arange(1, 1+dim_n), explained_variance_ratio_cumsum_mean, yerr=explained_variance_ratio_cumsum_sem, label='cum. var. explained')

    # 0.8 scaling index
    value_of_interest = 0.98
    scaling_idx_08 = np.interp(value_of_interest, scaling_idx_mean[::-1], np.arange(1, 1+dim_n)[::-1])
    cum_var_08 = np.interp(scaling_idx_08, np.arange(1, 1+dim_n), explained_variance_ratio_cumsum_mean)

    ax.plot([1, scaling_idx_08], [value_of_interest, value_of_interest], '--', color='black')
    ax.plot([scaling_idx_08, scaling_idx_08], [cum_var_08, value_of_interest], '--', color='black')
    ax.plot([1, scaling_idx_08], [cum_var_08, cum_var_08], '--', color='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Scaling components', fontsize=fs)
    plt.xticks(np.arange(1, 1+dim_n, 2), fontsize=fs)
    ax.set_ylim([0, 1.2])
    plt.yticks(np.arange(0, 1.2, 0.2), fontsize=fs)
    fig.legend(loc='upper left', fontsize=fs, frameon=False)

    plt.gca().annotate(value_of_interest, xy=[1, value_of_interest-0.02], xycoords="data", va="top", ha="center", fontsize=fs)
    plt.gca().annotate(f'{cum_var_08:.2f}', xy=[1, cum_var_08], xycoords="data", va="bottom", ha="center", fontsize=fs)

    plt.show()
    return fig


def velocity_in_scaling_space(serial_idxes, epoch='go', dim_n=9, dly_interval=1600,  scaling_idx_threshold=0.8, velocity_dim_n=3, noise_on=False):
    time_interval_prod_list = list()
    velocity_list = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        prod_intervals = np.array([600, 700, 800, 900, 1000, 1100, 1200])

        batch_size = len(prod_intervals)
        dly_intervals = np.array([dly_interval] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        stim1_off, stim2_on = trial_input.epochs[epoch]

        if epoch == 'go':
            outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
            motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
            stim2_on = stim1_off + motion_time

        firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))

        pca_obj = PCASpace.PCASpace(firing_rate_list, dim_n, model_dir)

        total_explained_variance_ratio = np.sum(pca_obj.pca.explained_variance_ratio_)

        pca_space_data = pca_obj.result  # tools.load_pickle(os.path.join(model_dir, 'pca_space' + '.pkl'))
        data_list = pca_space_data['data_pca']

        scaling_dir_name = os.path.join(model_dir, 'scaling_space' + '.pkl')

        ScalingSpaceObj = ScalingSpace.ScalingSpace(data_list, scaling_dir_name)

        scaling_dir_name = os.path.join(model_dir, 'scaling_space' + '.pkl')

        scaling_space_data = ScalingSpaceObj.result  # tools.load_pickle(scaling_dir_name)

        data_rescale = scaling_space_data['data_rescale']
        scaling_transform_matrix = scaling_space_data['scaling_transform_matrix']

        parameter_n, ref_time_n, pca_n = data_rescale.shape

        variance_explained_ratio = ScalingSpace.variance_explained(np.reshape(data_rescale, (-1, pca_n)),
                                                                   scaling_transform_matrix)

        scaling_idx = np.zeros((dim_n,))
        for pca_idx in range(0, dim_n):
            scaling_idx[pca_idx] = ScalingSpace.scaling_index(data_rescale, scaling_transform_matrix[:, pca_idx])

        #print(scaling_idx)
        #high_scaling_idx = np.argwhere(scaling_idx>scaling_idx_threshold)
        high_scaling_idx = np.arange(velocity_dim_n)#np.array([0, 1, 2]) #np.argsort(scaling_idx)[-3:]
        #print('high_scaling_idx', high_scaling_idx)
        high_scaling_transform_matrix = scaling_transform_matrix[:, high_scaling_idx].squeeze()

        traj_in_high_scaling_space = [np.matmul(x, high_scaling_transform_matrix) for x in data_list]
        #print(traj_in_high_scaling_space[0].shape)
        distance = np.array([np.sum(np.sqrt(np.sum(np.diff(x, axis=0) ** 2, axis=1))) for x in traj_in_high_scaling_space])
        velocity = distance/(stim2_on-stim1_off)

        time_interval_prod = (stim2_on-stim1_off)*20
        time_interval_prod_list.append(time_interval_prod)
        velocity_list.append(velocity)

    time_interval_prod_list = np.concatenate([x[np.newaxis, :] for x in time_interval_prod_list], axis=0)
    velocity_list = np.concatenate([x[np.newaxis, :] for x in velocity_list], axis=0)
    time_interval_mean = np.mean(time_interval_prod_list, axis=0)
    velocity_mean = np.mean(velocity_list, axis=0)

    #time_interval_mean = np.quantile(time_interval_prod_list, 0.5, axis=0)
    #velocity_mean = np.quantile(velocity_list, 0.5, axis=0)

    time_interval_sem = np.std(time_interval_prod_list, axis=0)/np.sqrt(time_interval_prod_list.shape[0])
    velocity_sem = np.std(velocity_list, axis=0)/np.sqrt(velocity_list.shape[0])

    #time_interval_sem = [np.quantile(time_interval_prod_list, 0.5, axis=0)-np.quantile(time_interval_prod_list, 0.25, axis=0),
                       #np.quantile(time_interval_prod_list, 0.75, axis=0)-np.quantile(time_interval_prod_list, 0.5, axis=0)]
    #velocity_sem = [np.quantile(velocity_list, 0.5, axis=0)-np.quantile(velocity_list, 0.25, axis=0),
                       #np.quantile(velocity_list, 0.75, axis=0)-np.quantile(velocity_list, 0.5, axis=0)]

    mpl.rcParams['xtick.minor.size'] = 0
    mpl.rcParams['xtick.minor.width'] = 0
    mpl.rcParams['ytick.minor.size'] = 0
    mpl.rcParams['ytick.minor.width'] = 0

    fig = plt.figure(figsize=(2.5, 2.1))
    #ax = fig.add_axes([0.2, 0.22, 0.65, 0.75])
    ax = fig.add_axes([0.2, 0.22, 0.65*0.66, 0.75])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.errorbar(time_interval_mean, velocity_mean, xerr=time_interval_sem, yerr=velocity_sem)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel('Time interval (ms)', fontsize=fs)
    ax.set_ylabel('Velocity (a.u.)', fontsize=fs)

    plt.xticks([600,  900, 1200], [600, 900, 1200], fontsize=fs)
    plt.yticks([1.4,  1.8,  2.2], [1.4,  1.8, 2.2], fontsize=fs)
    plt.show()
    return fig


def activity_peak_order_plot(serial_idx=0, epoch='interval', func_activity_threshold=1.5):
    model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

    prod_intervals = np.array([1200])
    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

    stim1_off, stim2_on = trial_input.epochs[epoch]

    if epoch == 'go':
        outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
        motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
        stim2_on = stim1_off + motion_time

    batch_idx = 0
    data = run_result.firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :].detach().cpu().numpy()
    max_firing_rate = np.max(data, axis=0)
    pick_idx = np.argwhere(max_firing_rate > func_activity_threshold).squeeze()

    data = data[:, pick_idx]
    peak_time = np.argmax(data, axis=0)
    peak_order = np.argsort(peak_time, axis=0)
    data = data[:, peak_order]

    # normalize
    for i in range(0, data.shape[1]):
        data[:, i] = data[:, i] / np.max(data[:, i])

    X, Y = np.mgrid[0:data.shape[0]*20:20, 0:data.shape[1]]

    #fig = plt.figure(figsize=(2.5, 2.1))
    #ax = fig.add_axes([0.2, 0.19, 0.75, 0.75])

    #fig = plt.figure(figsize=(2.5, 2.4))
    #ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])

    fig = plt.figure(figsize=(2.5, 2.4))
    ax = fig.add_axes([0.35, 0.25, 0.5, 0.6])

    plt.gca().set_xlabel('Time (ms)', fontsize=fs)
    plt.gca().set_ylabel('Sorted\nNeuron', fontsize=fs)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    # Make the plot
    plt.pcolormesh(X, Y, data)

    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)

    cbar = fig.colorbar(m, aspect=8)

    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.set_title('Normalized\n activity', fontsize=fs)

    plt.show()
    return fig


def connection_peak_order_plot_batch(serial_idxes, epoch='interval', func_activity_threshold=1.5):

    result_key = list()
    result_value = list()

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'interval_production/' + str(serial_idx)

        prod_intervals = np.array([1200])
        batch_size = len(prod_intervals)
        dly_intervals = np.array([1600] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='interval_production', is_cuda=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals)

        stim1_off, stim2_on = trial_input.epochs[epoch]

        if epoch == 'go':
            outputs = run_result.outputs.detach().cpu().numpy()[:, :, 0]
            motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])
            stim2_on = stim1_off + motion_time

        batch_idx = 0
        data = run_result.firing_rate_binder[stim1_off[batch_idx]:stim2_on[batch_idx], batch_idx, :].detach().cpu().numpy()
        max_firing_rate = np.max(data, axis=0)
        pick_idx = np.argwhere(max_firing_rate > func_activity_threshold).squeeze()
        #print('func neuron number: ', len(pick_idx))
        data = data[:, pick_idx]
        peak_time = np.argmax(data, axis=0)
        peak_order = np.argsort(peak_time, axis=0)
        pick_idx = pick_idx[peak_order]

        weight_hh = runnerObj.model.weight_hh.detach().cpu().numpy()

        weight_hh_pick = weight_hh[pick_idx, :][:, pick_idx]

        diag_sum_dict = dict()
        for i in range(0, len(pick_idx)):
            for j in range(0, len(pick_idx)):
                diag_sum_dict[j - i] = 0

        for i in range(0, len(pick_idx)):
            for j in range(0, len(pick_idx)):
                diag_sum_dict[j - i] += weight_hh_pick[i, j]

        diag_sum_dict[0] = 0

        diag_sum_list = list()
        diag_sum_key = list()
        for i in sorted(diag_sum_dict.keys()):
            diag_sum_key.append(i)
            diag_sum_list.append(diag_sum_dict[i])

        result_key.append(np.array(diag_sum_key))
        result_value.append(np.array(diag_sum_list))

    min_length = np.min(np.array([x[-1] for x in result_key]))
    for result_idx in range(len(result_key)):
        delete_ele_index = np.argwhere(np.abs(result_key[result_idx]) > min_length)
        result_key[result_idx] = np.delete(result_key[result_idx], delete_ele_index)
        result_value[result_idx] = np.delete(result_value[result_idx], delete_ele_index)[np.newaxis, :]

    result_value = np.concatenate(result_value, axis=0)
    result_mean = np.mean(result_value, axis=0)
    result_sem = np.std(result_value, axis=0)#/np.sqrt(result_value.shape[0])

    zero_idx = np.argwhere(np.abs(result_key[0]) == 0)
    result_key[0] = np.delete(result_key[0], zero_idx)
    result_mean = np.delete(result_mean, zero_idx)
    result_sem = np.delete(result_sem, zero_idx)

    fig = plt.figure(figsize=(2.5, 2.1))
    #ax = fig.add_axes([0.3, 0.25, 0.7, 0.6])
    ax = fig.add_axes([0.35, 0.25, 0.5, 0.6])

    plt.plot(np.arange(len(result_mean)), result_mean, 'blue')
    plt.fill_between(np.arange(len(result_mean)), result_mean-result_sem, result_mean+result_sem)

    tick_x_idx = np.argwhere(((result_key[0]) == 1) | (np.abs(result_key[0]) % 10 == 0))
    tick_x = np.arange(len(result_mean))[tick_x_idx].flatten()
    tick_x_label = result_key[0][tick_x_idx].flatten()

    plt.xticks(tick_x, tick_x_label, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.gca().set_xlabel('Peak order difference', fontsize=fs)
    plt.gca().set_ylabel('Recurrent\nweight', fontsize=fs)

    plt.show()
    return fig


