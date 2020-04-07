from __future__ import division

import torch
import numpy as np
from matplotlib import pyplot as plt

from .. import task
from .. import default

import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm

import os
from scipy import stats
from numba import jit

from .. import run

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

plt.style.use('default')



class LeftAlignAvr:
    def __init__(self, max_length):
        self.value = np.zeros((max_length,))
        self.value_2 = np.zeros((max_length,))

        self.number = np.zeros((max_length,))
        self.max_length = max_length

    def add(self, add_value):
        temp_length = len(add_value)
        self.value[0:temp_length] += add_value
        self.value_2[0:temp_length] += add_value * add_value

        self.number[0:temp_length] += 1

    def return_average(self):
        zero_number_idx = np.argwhere(self.number <= 1)
        self.number = np.delete(self.number, zero_number_idx)
        self.value = np.delete(self.value, zero_number_idx)
        self.value_2 = np.delete(self.value_2, zero_number_idx)

        return self.value/self.number

    def return_sem(self):
        zero_number_idx = np.argwhere(self.number <= 1)
        self.number = np.delete(self.number, zero_number_idx)
        self.value = np.delete(self.value, zero_number_idx)
        self.value_2 = np.delete(self.value_2, zero_number_idx)

        return np.sqrt((self.value_2/self.number)-(self.value/self.number)**2)/np.sqrt(self.number)


class RightAlignAvr:
    def __init__(self, max_length):
        self.value = np.zeros((max_length,))
        self.value_2 = np.zeros((max_length,))

        self.number = np.zeros((max_length,))
        self.max_length = max_length

    def add(self, add_value):
        temp_length = len(add_value)
        self.value[self.max_length-temp_length:self.max_length] += add_value
        self.value_2[self.max_length-temp_length:self.max_length] += add_value * add_value

        self.number[self.max_length-temp_length:self.max_length] += 1

    def return_average(self):
        zero_number_idx = np.argwhere(self.number <= 1)
        self.number = np.delete(self.number, zero_number_idx)
        self.value = np.delete(self.value, zero_number_idx)
        self.value_2 = np.delete(self.value_2, zero_number_idx)

        return self.value/self.number

    def return_sem(self):
        zero_number_idx = np.argwhere(self.number <= 1)

        self.number = np.delete(self.number, zero_number_idx)
        self.value = np.delete(self.value, zero_number_idx)
        self.value_2 = np.delete(self.value_2, zero_number_idx)

        return np.sqrt((self.value_2/self.number)-(self.value/self.number)**2)/np.sqrt(self.number)


@jit(nopython=True)
def local_maximum_location(signal, loc):
    if loc == 0:
        for x in range(len(signal)-1):
            if signal[x] >= signal[x+1]:
                return x
        return len(signal)-1
    elif loc == len(signal)-1:
        for x in range(len(signal)-1, 0, -1):
            if signal[x-1] < signal[x]:
                return x
        return 0
    elif signal[loc-1] < signal[loc]:
        for x in range(loc, len(signal) - 1):
            if signal[x] >= signal[x + 1]:
                return x
        return len(signal) - 1
    else:
        for x in range(loc-1, 0, -1):
            if signal[x-1] < signal[x]:
                return x
        return 0


@jit(nopython=True)
def no_greater_than_loc(sorted_array, value):
    for i in range(len(sorted_array)):
        if sorted_array[i] >= value:
            return i
    return len(sorted_array)


#@jit(nopython=True)
def p_n_neuron_split(output_neurons, input_neuron, local_peak_outputs, local_peak_input):
    index_of_duplicate = np.argwhere(output_neurons==input_neuron)

    if local_peak_outputs[-1] < local_peak_input:
        delim = len(local_peak_outputs)
    else:
        for i in range(len(local_peak_outputs)):
            if local_peak_outputs[i] >= local_peak_input:
                delim = i
                break

    if len(index_of_duplicate) == 0:
        return np.arange(delim, len(output_neurons)), np.arange(0, delim)
    else:
        if index_of_duplicate[0,0]>=delim:
            return np.concatenate([np.arange(delim, index_of_duplicate[0,0]), np.arange(index_of_duplicate[0,0]+1, len(output_neurons))]), np.arange(0, delim)
        else:#index_of_duplicate[0,0]<delim
            return np.arange(delim, len(output_neurons)), np.concatenate([np.arange(0, index_of_duplicate[0,0]), np.arange(index_of_duplicate[0,0]+1, delim)])


class WeightConnection:
    def __init__(self, stim_n, func_relevance_threshold=2.):
        '''
        :param firing_rate_binder: shape (stim, time, neuron)
        :param weight_hh: shape (input_n, output_n)
        :param time_points: shape (time_n,)
        '''
        self.stim_n = stim_n
        self.func_relevance_threshold = func_relevance_threshold
        self.p_result_list = [LeftAlignAvr(100) for i in range(self.stim_n)]
        self.n_result_list = [RightAlignAvr(100) for i in range(self.stim_n)]

    def reset_parameter(self, firing_rate_binder, weight_hh, time_points):
        self.firing_rate_binder = firing_rate_binder.astype(np.float64)
        self.weight_hh = weight_hh
        self.time_points = time_points

        self.func_relevant_neuron = firing_rate_binder>self.func_relevance_threshold

    def weight_connection(self, input_neurons_1, output_neurons_1, local_peak_inputs, local_peak_outputs, stim_idx1, stim_idx2):
        order_idx_inputs = np.argsort(local_peak_inputs)
        order_idx_outputs = np.argsort(local_peak_outputs)
        local_peak_inputs = local_peak_inputs[order_idx_inputs]
        local_peak_outputs = local_peak_outputs[order_idx_outputs]
        input_neurons = input_neurons_1[order_idx_inputs]
        output_neurons = output_neurons_1[order_idx_outputs]

        if len(output_neurons) == 0:
            return

        weight = self.weight_hh[input_neurons, :][:, output_neurons]
        #print(weight.shape)
        stim_diff = np.abs(stim_idx1-stim_idx2)
        for i in range(len(local_peak_inputs)):

            p_neuron, n_neuron = p_n_neuron_split(output_neurons, input_neurons[i], local_peak_outputs, local_peak_inputs[i])

            self.p_result_list[stim_diff].add(weight[i, p_neuron])
            self.n_result_list[stim_diff].add(weight[i, n_neuron])

    def do(self):
        for time_point in self.time_points:

            func_relevance_neuron = np.argwhere(np.sum(self.func_relevant_neuron[:, time_point, :], axis=0) > 0).flatten()

            firing_rate = self.firing_rate_binder[:, time_point, func_relevance_neuron]
            max_stim = np.argmax(firing_rate, axis=0)
            func_relevant_neuron_single_time = [func_relevance_neuron[np.argwhere(max_stim==stim_idx).flatten()] for stim_idx in range(self.stim_n)]
            #func_relevant_neuron_single_time = [np.argwhere(self.func_relevant_neuron[stim_idx, time_point, :]).squeeze() for stim_idx in range(self.stim_n)]

            for stim_idx1 in range(self.stim_n):
                input_neurons = func_relevant_neuron_single_time[stim_idx1]
                #print(input_neurons)
                #print([x for x in input_neurons])
                local_peak_inputs = np.array([local_maximum_location(self.firing_rate_binder[stim_idx1, :, x], time_point) for x in input_neurons])  # peak location when c = 0.01

                for stim_idx2 in range(stim_idx1, self.stim_n):
                    output_neurons = func_relevant_neuron_single_time[stim_idx2]
                    local_peak_outputs = np.array([local_maximum_location(self.firing_rate_binder[stim_idx2, :, x], time_point) for x in output_neurons])  # peak location when c = 0.01
                    self.weight_connection(input_neurons, output_neurons, local_peak_inputs, local_peak_outputs, stim_idx1, stim_idx2)

    def return_mean(self):
        delete_num = 1
        p_mean = [x.return_average()[:-delete_num] for x in self.p_result_list]
        n_mean = [x.return_average()[delete_num:] for x in self.n_result_list]
        return p_mean, n_mean

    def return_sem(self):
        delete_num = 1
        p_sem = [x.return_sem()[:-delete_num] for x in self.p_result_list]
        n_sem = [x.return_sem()[delete_num:] for x in self.n_result_list]
        return p_sem, n_sem


def weight_connection_with_time(serial_idxes, epoch='interval', time_points=np.array([600]), func_relevance_threshold=2., noise_on=False):

    ring_centers = np.arange(6., 25., 2.) #np.array([6., 12., 18., 24.])

    weight_connection_obj = WeightConnection(len(ring_centers), func_relevance_threshold=func_relevance_threshold)

    time_points = (time_points/20).astype(np.int)

    dly_interval = 1600
    prod_intervals = np.array([1200])

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)

    dly_intervals = np.array([dly_interval] * batch_size)

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gaussian_center=ring_centers)

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        firing_rate_binder = np.concatenate(list(firing_rate_binder[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)), axis=0)

        weight_connection_obj.reset_parameter(firing_rate_binder, runnerObj.model.weight_hh.detach().cpu().numpy(), time_points)
        weight_connection_obj.do()

    p_mean, n_mean = weight_connection_obj.return_mean()

    xmax = 8
    ymin = -6
    ymax = 6
    x, y = np.mgrid[range(0, int(xmax/2+1+1)), range(ymin, ymax+1+1)]
    result = np.zeros((len(range(0, int(xmax/2+1))), len(range(ymin, ymax+1))))

    for i in range(len(p_mean)):
        t_coord_total = np.concatenate([-np.flip(np.arange(len(n_mean[i])), axis=0)-1, np.arange(len(p_mean[i]))])
        mean_total = np.concatenate([n_mean[i], p_mean[i]])
        stim_coord = ring_centers[i]-ring_centers[0]

        if stim_coord <= xmax:
            for y1, z1 in zip(t_coord_total, mean_total):
                if y1<=ymax and y1>=ymin:
                    result[int(stim_coord/2), y1-ymin] = z1

            if t_coord_total[-1] < ymax:
                value = result[int(stim_coord/2), t_coord_total[-1]-ymin]
                for i in range(t_coord_total[-1], ymax+1):
                    result[int(stim_coord/2), i-ymin] = value
            if t_coord_total[0] > ymin:
                value = result[int(stim_coord/2), t_coord_total[0]-ymin]
                for i in range(ymin, t_coord_total[0]):
                    result[int(stim_coord/2), i-ymin] = value

    zmin = np.abs(np.min(result))
    zmax = np.max(result)

    norm = mpl.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "darkblue"], [norm((zmin-zmax)/(zmax+zmin)), "white"], [norm(1.0), "red"]]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)

    fig = plt.figure(figsize=(2.5, 2.3))
    ax = fig.add_axes([0.23, 0.2, 0.63, 0.63])

    plt.pcolormesh(x, y, result, cmap=cmap)
    plt.xticks([0+0.5, 2+0.5, 4+0.5], [0, 4, 8], fontsize=fs)
    plt.yticks([-6+0.5, -1+0.5, 0+0.5, 6+0.5], [-6, -1, 1, 7], fontsize=fs)
    plt.gca().set_xlabel('$|x_1-x_2|$', fontsize=fs)
    plt.gca().set_ylabel('Peak order difference', fontsize=fs)

    # color bar
    #cmap = mpl.rcParams["image.cmap"]  # plt.get_cmap('rainbow')

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    m.set_clim(vmin=np.min(result), vmax=np.max(result))

    cbar = fig.colorbar(m)
    #cbar.set_ticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.set_title('Recurrent\n weight', fontsize=fs)

    plt.show()
    return fig


def data_rescale_fcn(data_pca):

    # rescale time of data_pca
    maxT = max([x.shape[0] for x in data_pca])
    ref_time = np.array(range(maxT)) / (maxT - 1)

    # dimension: (parameter_n, ref_time, pca_n)
    data_rescale = np.zeros((len(data_pca), ref_time.shape[0], data_pca[0].shape[1]))

    for para_idx in range(len(data_pca)):
        x = data_pca[para_idx]
        curr_time = np.array(range(x.shape[0])) / (x.shape[0] - 1)
        for pca_idx in range(x.shape[1]):
            data_rescale[para_idx, :, pca_idx] = np.interp(ref_time, curr_time, x[:, pca_idx])

    return data_rescale


def firing_rate_compare(serial_idx=0, epoch='interval', mode='pilot'):
    model_dir = './core/model/'+'timed_spatial_reproduction/'+str(serial_idx)

    ring_centers = np.array([6., 12., 18., 24.])
    colors = np.array(['#A6BDD7', '#C10020', '#803E75', '#FFB300'])

    batch_size = len(ring_centers)

    prod_intervals = np.array([1200] * batch_size)
    dly_intervals = np.array([1600] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

    stim1_off, stim2_on = trial_input.epochs[epoch]

    firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))
    mean_firing_rate_list = list(np.expand_dims(np.mean(x, axis=0), axis=0) for x in firing_rate_list)

    mean_firing_rate = np.mean(np.concatenate(mean_firing_rate_list, axis=0), axis=0)
    #print(np.argsort(mean_firing_rate).shape)
    mean_firing_rate_order = np.flip(np.argsort(mean_firing_rate),axis=0)

    if mode == 'pilot':
        # pilot test
        plot_number = 16
        for plot_idx in range(0, plot_number):
            neuron_idx = mean_firing_rate_order[plot_idx]
            plt.figure(figsize=(2.5, 2.1))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                plt.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i], color=colors[i])
            plt.show()
    else:
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(2.5, 2.1))
        if epoch=='interval':
            # 0, 1, 4, 8

            plot_idx = 4
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax0.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i], color=colors[i])

            plot_idx = 1
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax1.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i], color=colors[i])

            plot_idx = 3
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax2.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i], color=colors[i])

            plot_idx = 0
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax3.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i], color=colors[i])
        elif epoch == 'go':
            # 0, 1, 2, 8

            plot_idx = 0
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax0.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i],
                         color=colors[i])

            plot_idx = 1
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax1.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i],
                         color=colors[i])

            plot_idx = 2
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax2.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i],
                         color=colors[i])

            plot_idx = 8
            neuron_idx = mean_firing_rate_order[plot_idx]
            firing_rate_list_plot = list(x[:, neuron_idx] for x in firing_rate_list)
            for i in range(0, len(firing_rate_list_plot)):
                ax3.plot(np.array(range(0, len(firing_rate_list_plot[i]))) * 20, firing_rate_list_plot[i],
                         color=colors[i])

        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        ax2.set_ylim(bottom=-5)
        ax2.set_xlim(left=-70)

        ax0.set_xlim(left=-70)
        ax1.set_xlim(left=-70)
        ax3.set_xlim(left=-70)

        ax2.plot([0, 500], [-2, -2], color='black')
        ax2.plot([-50, -50], [0, 20], color='black')

        plt.show()
        return f


def PCA_2d_plot_with_time_stim_direction(serial_idx=0, prod_intervals=np.array([1200]), epoch='interval', noise_on=False):
    model_dir = './core/model/'+'timed_spatial_reproduction/'+str(serial_idx)

    #ring_centers = np.arange(6., 32. - 6., 2.)

    alphas = np.arange(1, len(prod_intervals)+1)/len(prod_intervals)

    ring_centers = np.array([6., 12., 18., 24.])
    colors = np.array([0, 1, 2, 3])
    color_dict = {0:'#A6BDD7', 1:'#C10020', 2:'#803E75', 3:'#FFB300'}
    #colors = np.array(['#A6BDD7', '#C10020', '#803E75', '#FFB300'])

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    alphas, colors = np.meshgrid(alphas, colors)
    alphas = alphas.flatten()
    colors = colors.flatten()

    colors = [color_dict[x] for x in colors]

    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)

    runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

    stim1_off, stim2_on = trial_input.epochs[epoch]

    #if epoch == 'delay':
    #    stim1_off = stim2_on - 1
    stim1_off = stim1_off + 20
    # (stimulus, time, neuron)
    firing_rate_list = np.concatenate(list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :,  :] for i in range(0, batch_size)), axis=0)

    concate_firing_rate = np.reshape(firing_rate_list, (-1, firing_rate_list.shape[-1]))

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)
    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    time_size = stim2_on - stim1_off

    delim = np.cumsum(time_size)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)

    ##########################################################################################

    #component of time

    concate_firing_rate_time = np.mean(firing_rate_list, axis=0)

    pca_time = PCA(n_components=1)
    pca_time.fit(concate_firing_rate_time)

    ##########################################################################################
    #component of stimulus

    concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)

    pca_stim = PCA(n_components=1)
    pca_stim.fit(concate_firing_rate_stim)

    ##########################################################################################

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax = fig.gca()
    for i in range(0, len(concate_transform_split)):
        #print(colors[i], alphas[i])
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], color=colors[i], alpha=alphas[i])
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], marker='*', color=colors[i], alpha=alphas[i])
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], marker='o', color=colors[i], alpha=alphas[i])
    ax.plot([-30*np.sum(pca.components_[0]*pca_time.components_[0]), 50*np.sum(pca.components_[0]*pca_time.components_[0])], [-30*np.sum(pca.components_[1]*pca_time.components_[0]), 50*np.sum(pca.components_[1]*pca_time.components_[0])], '--', color='black')
    ax.plot([-60*np.sum(pca.components_[0]*pca_stim.components_[0]), 65*np.sum(pca.components_[0]*pca_stim.components_[0])], [-60*np.sum(pca.components_[1]*pca_stim.components_[0]), 65*np.sum(pca.components_[1]*pca_stim.components_[0])], '--', color='black')

    ax.annotate('f-PC1', xy=(50*np.sum(pca.components_[0]*pca_time.components_[0]), 50*np.sum(pca.components_[1]*pca_time.components_[0])),
                xytext=(50*np.sum(pca.components_[0]*pca_time.components_[0]), 50*np.sum(pca.components_[1]*pca_time.components_[0])), xycoords='data',
                fontsize=fs, ha='center', va='bottom')
    '''
    ax.annotate('s-PC1', xy=(71*np.sum(pca.components_[0]*pca_stim.components_[0]), 71*np.sum(pca.components_[1]*pca_stim.components_[0])),
                xycoords='data',
                fontsize=fs, ha='center', va='bottom')
    '''
    ax.annotate('s-PC1', xy=(40*np.sum(pca.components_[0]*pca_stim.components_[0]), 40*np.sum(pca.components_[1]*pca_stim.components_[0])),
                xycoords='data',
                fontsize=fs, ha='center', va='bottom')

    ax.set_xlabel('PC1', fontsize=fs)
    ax.set_ylabel('PC2', fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    plt.show()
    return fig


def orthogonality_perception(serial_idxes):
    '''
    Dot product between 1st PCs
    '''
    result_interval = list()
    #result_go = list()
    transient_period = 20
    ring_centers = np.arange(6., 32.-6, 2.)#np.array([6., 12., 18., 24.])

    batch_size = len(ring_centers)
    prod_intervals = np.array([1200] * batch_size)
    dly_intervals = np.array([1600] * batch_size)

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        ##########################################################
        stim1_off, stim2_on = trial_input.epochs['interval']
        stim1_off = stim1_off + transient_period

        # (stimulus, time, neuron)
        firing_rate_list = np.concatenate(list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)), axis=0)

        # component of time
        concate_firing_rate_time = np.mean(firing_rate_list, axis=0)
        pca_time = PCA(n_components=2)
        pca_time.fit(concate_firing_rate_time)

        # component of stimulus
        concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)
        pca_stim = PCA(n_components=2)
        pca_stim.fit(concate_firing_rate_stim)

        result_interval.append(180/np.pi*np.arccos(np.abs(np.sum(pca_time.components_[0]*pca_stim.components_[0]))))

    #fig1 = plt.figure(figsize=(1.2, 2.1))
    #ax = fig1.add_axes([0.4, 0.4, 0.5, 0.5])
    fig1 = plt.figure(figsize=(1., 2.1))
    ax = fig1.add_axes([0.4, 0.4, 0.45, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #sns.distplot(np.array(result_interval), kde=False, norm_hist=True, bins=np.arange(np.pi/4, np.pi/2+np.pi/4/8, np.pi/4/8), color='magenta', hist_kws=dict(alpha=1))
    sns.distplot(np.array(result_interval), kde=False, norm_hist=True, color='magenta', hist_kws=dict(alpha=1))
    ymax = ax.get_ylim()[1]

    plt.xlabel('Angle\n(deg)', fontsize=fs)
    plt.ylabel('P.D.F.', fontsize=fs)
    plt.xticks([45, 90], [45, 90], fontsize=fs)
    plt.plot([45, 45], [0, ymax], '--', color='black')

    if stats.ttest_1samp(result_interval, 45)[1] < 0.05:
        ax.annotate('*', xy=(45, ymax), xycoords='data', fontsize=fs)
    '''
    plt.xticks([np.pi/4, np.pi/2], ['$\pi$/4', '$\pi$/2'], fontsize=fs)
    plt.plot([np.pi/4, np.pi/4], [0, ymax], '--', color='black')

    if stats.ttest_1samp(result_interval, np.pi/4)[1] < 0.05:
        ax.annotate('*', xy=(np.pi/4, ymax), xycoords='data', fontsize=fs)
    '''
    plt.yticks(fontsize=fs)
    plt.show()

    return fig1


def portion_of_variance_perception(serial_idxes):
    '''
    portion of variance in perception epoch
    '''
    space_portion = list()
    time_portion = list()
    mix_portion = list()

    ring_centers = np.arange(6., 32.-6, 2.)#np.array([6., 12., 18., 24.])

    batch_size = len(ring_centers)
    prod_intervals = np.array([1200] * batch_size)
    dly_intervals = np.array([1600] * batch_size)

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        ##########################################################
        stim1_off, stim2_on = trial_input.epochs['interval']
        stim1_off = stim1_off + 20

        # (stimulus, time, neuron)
        firing_rate_list = np.concatenate(list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)), axis=0)

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(0, 1))[np.newaxis, np.newaxis, :]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean

        firing_rate_list_space = np.mean(firing_rate_list_centered, axis=1)
        var_space = np.var(firing_rate_list_space)/np.var(firing_rate_list_centered)
        space_portion.append(var_space)

        firing_rate_list_interval = np.mean(firing_rate_list_centered, axis=0)
        var_interval = np.var(firing_rate_list_interval)/np.var(firing_rate_list_centered)
        time_portion.append(var_interval)
        ##############################
        mix_portion.append(1-var_space-var_interval)

    space_portion = np.mean(np.array(space_portion))
    time_portion = np.mean(np.array(time_portion))
    mix_portion = np.mean(np.array(mix_portion))


    # plot
    fig = plt.figure(figsize=(2, 2.1))
    ax = fig.add_axes([0.4, 0.2, 0.1, 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    barWidth = 0.5
    r = 0
    #names = ['SR', 't-SR', 'DM', 't-DM']
    # Create orange Bars
    plt.bar(r, space_portion, color='#f9bc86', edgecolor='white', width=barWidth*0.92,
            label="S"+": "+str(round(100*space_portion, 1))+"%")
    # Create green Bars
    plt.bar(r, time_portion, bottom=space_portion, color='#b5ffb9', edgecolor='white', width=barWidth*0.92,
            label="F"+": "+str(round(100*time_portion, 1))+"%")
    # Create blue Bars
    plt.bar(r, mix_portion, bottom=time_portion+space_portion, color='#a3acff', edgecolor='white',
            width=barWidth*0.92, label="S+F"+": "+str(round(100*mix_portion, 1))+"%")
    plt.ylabel('Portion of\nvariance', fontsize=fs)

    # Custom x axis
    plt.xticks([], [])
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    #red_dot, = plt.plot(z, "ro", markersize=15)
    plt.legend(loc=(-3.6, 0.95), fontsize=fs, frameon=False, ncol=1, handlelength=0.7)

    # Show graphic
    plt.show()

    return fig


def orthogonality_delay(serial_idxes):
    '''
    Dot product between 1st PCs
    '''
    result_delay = list()

    prod_intervals = np.array([600, 700, 800, 900, 1000, 1100, 1200])
    ring_centers = np.arange(6., 32.-6, 2.)

    prod_intervals_num = len(prod_intervals)
    ring_centers_num = len(ring_centers)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)


    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'delay'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        stim1_off = stim2_on - 1

        firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))

        concate_firing_rate = np.concatenate(firing_rate_list, axis=0)
        firing_rate_list = concate_firing_rate.reshape(ring_centers_num, prod_intervals_num, concate_firing_rate.shape[-1])

        ##########################################################################################
        # component of time
        concate_firing_rate_time = np.mean(firing_rate_list, axis=0)

        pca_time = PCA(n_components=2)
        pca_time.fit(concate_firing_rate_time)

        ##########################################################################################
        # component of stimulus
        concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)

        pca_stim = PCA(n_components=2)
        pca_stim.fit(concate_firing_rate_stim)
        ##########################################################################################

        result_delay.append(180/np.pi*np.arccos(np.abs(np.sum(pca_time.components_[0]*pca_stim.components_[0]))))

    fig1 = plt.figure(figsize=(1., 2.1))
    ax = fig1.add_axes([0.4, 0.4, 0.45, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #sns.distplot(np.array(result_delay), kde=False, norm_hist=True, bins=np.arange(-1, 1, 0.2), color='magenta', hist_kws=dict(alpha=1))
    sns.distplot(np.array(result_delay), kde=False, norm_hist=True, color='magenta', hist_kws=dict(alpha=1))

    ymax = ax.get_ylim()[1]

    plt.xlabel('Angle\n(deg)', fontsize=fs)
    plt.ylabel('P.D.F.', fontsize=fs)
    plt.xticks([45, 90], [45, 90], fontsize=fs)
    plt.plot([45, 45], [0, ymax], '--', color='black')

    if stats.ttest_1samp(result_delay, 45)[1] < 0.05:
        ax.annotate('*', xy=(45, ymax), xycoords='data', fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.show()

    return fig1


def portion_of_variance_delay(serial_idxes):
    '''
    Dot product between 1st PCs
    '''
    space_portion = list()
    time_portion = list()
    mix_portion = list()

    prod_intervals = np.array([600, 700, 800, 900, 1000, 1100, 1200])
    ring_centers = np.arange(6., 32.-6, 2.)

    prod_intervals_num = len(prod_intervals)
    ring_centers_num = len(ring_centers)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)


    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        epoch = 'delay'
        stim1_off, stim2_on = trial_input.epochs[epoch]

        stim1_off = stim2_on - 1

        firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))

        concate_firing_rate = np.concatenate(firing_rate_list, axis=0)
        firing_rate_list = concate_firing_rate.reshape(ring_centers_num, prod_intervals_num, concate_firing_rate.shape[-1])

        firing_rate_list_mean = np.mean(firing_rate_list, axis=(0, 1))[np.newaxis, np.newaxis, :]
        firing_rate_list_centered = firing_rate_list - firing_rate_list_mean

        firing_rate_list_space = np.mean(firing_rate_list_centered, axis=1)
        var_space = np.var(firing_rate_list_space)/np.var(firing_rate_list_centered)
        space_portion.append(var_space)

        firing_rate_list_interval = np.mean(firing_rate_list_centered, axis=0)
        var_interval = np.var(firing_rate_list_interval)/np.var(firing_rate_list_centered)
        time_portion.append(var_interval)
        ##############################
        mix_portion.append(1-var_space-var_interval)

    space_portion = np.mean(np.array(space_portion))
    time_portion = np.mean(np.array(time_portion))
    mix_portion = np.mean(np.array(mix_portion))

    # plot
    fig = plt.figure(figsize=(2, 2.1))
    ax = fig.add_axes([0.4, 0.2, 0.1, 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    barWidth = 0.5
    r = 0
    #names = ['SR', 't-SR', 'DM', 't-DM']
    # Create orange Bars
    plt.bar(r, space_portion, color='#f9bc86', edgecolor='white', width=barWidth*0.92,
            label="S"+": "+str(round(100*space_portion, 1))+"%")
    # Create green Bars
    plt.bar(r, time_portion, bottom=space_portion, color='#b5ffb9', edgecolor='white', width=barWidth*0.92,
            label="I"+": "+str(round(100*time_portion, 1))+"%")
    # Create blue Bars
    plt.bar(r, mix_portion, bottom=time_portion+space_portion, color='#a3acff', edgecolor='white',
            width=barWidth*0.92, label="S+I"+": "+str(round(100*mix_portion, 1))+"%")
    plt.ylabel('Portion of\nvariance', fontsize=fs)

    # Custom x axis
    plt.xticks([], [])
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    #red_dot, = plt.plot(z, "ro", markersize=15)
    plt.legend(loc=(-3.6, 0.95), fontsize=fs, frameon=False, ncol=1, handlelength=0.7)

    # Show graphic
    plt.show()

    return fig


def orthogonality_production(serial_idxes):
    '''
    Dot product between 1st PCs
    '''
    prod_intervals_1 = np.arange(600, 1220, 40)
    ring_centers_1 = np.arange(6., 32. - 6, 2.)

    num_ring_center = len(ring_centers_1)
    num_prod_interval = len(prod_intervals_1)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals_1, ring_centers_1)

    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)

    angle_space_tf = list()
    angle_space_ti = list()
    angle_tf_ti = list()

    #########

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

        ##########################################################
        epoch = 'go'
        stim1_off, stim2_on = trial_input.epochs[epoch]
        stim1_off = stim1_off #+ 10

        # (stimulus, time, neuron)
        firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))
        firing_rate_rescale = data_rescale_fcn(firing_rate_list)
        firing_rate_rescale = firing_rate_rescale[:, 10:, :]
        # (space, interval, time, neuron)
        firing_rate_list = firing_rate_rescale.reshape(num_ring_center, num_prod_interval, firing_rate_rescale.shape[-2], firing_rate_rescale.shape[-1])

        ##########################################################################################
        # component of space
        concate_firing_rate_space = np.mean(firing_rate_list, axis=(1,2))

        pca_space = PCA(n_components=2)
        pca_space.fit(concate_firing_rate_space)
        ##########################################################################################
        # component of interval
        concate_firing_rate_ti = np.mean(firing_rate_list, axis=(0,2))

        pca_ti = PCA(n_components=2)
        pca_ti.fit(concate_firing_rate_ti)
        ##########################################################################################
        # component of temporal flow
        concate_firing_rate_tf = np.mean(firing_rate_list, axis=(0,1))

        pca_tf = PCA(n_components=2)
        pca_tf.fit(concate_firing_rate_tf)
        ##########################################################################################
        angle_space_tf.append(180/np.pi*np.arccos(np.abs(np.sum(pca_space.components_[0]*pca_tf.components_[0]))))
        angle_space_ti.append(180/np.pi*np.arccos(np.abs(np.sum(pca_space.components_[0]*pca_ti.components_[0]))))
        angle_tf_ti.append(180/np.pi*np.arccos(np.abs(np.sum(pca_ti.components_[0]*pca_tf.components_[0]))))

    fig1 = plt.figure(figsize=(1., 2.1))
    ax = fig1.add_axes([0.4, 0.4, 0.45, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.distplot(np.array(angle_space_tf), kde=False, norm_hist=True, color='magenta', hist_kws=dict(alpha=1))

    ymax = ax.get_ylim()[1]

    plt.xlabel('Angle\n(deg)', fontsize=fs)
    plt.ylabel('P.D.F.', fontsize=fs)
    plt.xticks([45, 90], [45, 90], fontsize=fs)
    plt.yticks([0, 0.05],  fontsize=fs)

    plt.plot([45, 45], [0, ymax], '--', color='black')

    if stats.ttest_1samp(angle_space_tf, 45)[1] < 0.05:
        ax.annotate('*', xy=(45, ymax), xycoords='data', fontsize=fs)

    plt.yticks(fontsize=fs)

    ###############################
    fig2 = plt.figure(figsize=(1., 2.1))
    ax = fig2.add_axes([0.4, 0.4, 0.45, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.distplot(np.array(angle_space_ti), kde=False, norm_hist=True, color='magenta', hist_kws=dict(alpha=1))

    ymax = ax.get_ylim()[1]

    plt.xlabel('Angle\n(deg)', fontsize=fs)
    #plt.ylabel('P.D.F.', fontsize=fs)
    plt.xticks([45, 90], [45, 90], fontsize=fs)
    plt.plot([45, 45], [0, ymax], '--', color='black')

    if stats.ttest_1samp(angle_space_tf, 45)[1] < 0.05:
        ax.annotate('*', xy=(45, ymax), xycoords='data', fontsize=fs)

    plt.yticks(fontsize=fs)
    #################################
    fig3 = plt.figure(figsize=(1., 2.1))
    ax = fig3.add_axes([0.4, 0.4, 0.45, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.distplot(np.array(angle_tf_ti), kde=False, norm_hist=True, color='magenta', hist_kws=dict(alpha=1))

    ymax = ax.get_ylim()[1]
    ax.set_xlim([0, 90])
    plt.xlabel('Angle\n(deg)', fontsize=fs)
    #plt.ylabel('P.D.F.', fontsize=fs)
    plt.xticks([0, 45, 90], [0, 45, 90], fontsize=fs)
    plt.plot([45, 45], [0, ymax], '--', color='black')

    #if stats.ttest_1samp(angle_tf_ti, np.pi/4)[1] < 0.05:
    #    ax.annotate('*', xy=(np.pi/4, ymax), xycoords='data', fontsize=fs)

    plt.yticks(fontsize=fs)

    plt.show()

    return fig1, fig2, fig3


def portion_of_variance_production(serial_idxes, mode="calculate"):
    '''
    Dot product between 1st PCs
    '''
    prod_intervals_1 = np.arange(600, 1220, 40)
    ring_centers_1 = np.arange(6., 32. - 6, 2.)

    num_ring_center = len(ring_centers_1)
    num_prod_interval = len(prod_intervals_1)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals_1, ring_centers_1)

    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)
    dly_intervals = np.array([1600] * batch_size)

    time_portion = list()
    interval_portion = list()
    space_portion = list()
    mix_portion = list()
    time_interval_portion = list()
    interval_space_portion = list()
    time_space_portion = list()
    mix_3_portion = list()
    #########
    if mode == "calculate":
        for serial_idx in serial_idxes:

            model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

            if not os.path.exists(model_dir):
                continue

            runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=False)
            trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals, gaussian_center=ring_centers)

            firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()

            ##########################################################
            epoch = 'go'
            stim1_off, stim2_on = trial_input.epochs[epoch]
            stim1_off = stim1_off #+ 10

            # (stimulus, time, neuron)
            firing_rate_list = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))
            firing_rate_rescale = data_rescale_fcn(firing_rate_list)
            firing_rate_rescale = firing_rate_rescale[:, 10:, :]
            # (space, interval, time, neuron)
            firing_rate_list = firing_rate_rescale.reshape(num_ring_center, num_prod_interval, firing_rate_rescale.shape[-2], firing_rate_rescale.shape[-1])

            firing_rate_list_mean = np.mean(firing_rate_list, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]
            firing_rate_list_centered = firing_rate_list - firing_rate_list_mean

            firing_rate_list_space = np.mean(firing_rate_list_centered, axis=(1, 2))
            var_space = np.var(firing_rate_list_space)/np.var(firing_rate_list_centered)
            space_portion.append(var_space)

            firing_rate_list_interval = np.mean(firing_rate_list_centered, axis=(0, 2))
            var_interval = np.var(firing_rate_list_interval)/np.var(firing_rate_list_centered)
            interval_portion.append(var_interval)

            firing_rate_list_time = np.mean(firing_rate_list_centered, axis=(0, 1))
            var_time = np.var(firing_rate_list_time)/np.var(firing_rate_list_centered)
            time_portion.append(var_time)

            firing_rate_list_mix = firing_rate_list_centered - firing_rate_list_space[:, np.newaxis, np.newaxis, :] - firing_rate_list_interval[np.newaxis, :, np.newaxis, :] - firing_rate_list_time[np.newaxis, np.newaxis, :, :]
            var_mix = np.var(firing_rate_list_mix)/np.var(firing_rate_list_centered)
            mix_portion.append(var_mix)

            firing_rate_list_space_interval = np.mean(firing_rate_list_mix, axis=2)
            var_space_interval = np.var(firing_rate_list_space_interval)/np.var(firing_rate_list_centered)
            interval_space_portion.append(var_space_interval)

            firing_rate_list_space_time = np.mean(firing_rate_list_mix, axis=1)
            var_space_time = np.var(firing_rate_list_space_time)/np.var(firing_rate_list_centered)
            time_space_portion.append(var_space_time)

            firing_rate_list_interval_time = np.mean(firing_rate_list_mix, axis=0)
            var_interval_time = np.var(firing_rate_list_interval_time)/np.var(firing_rate_list_centered)
            time_interval_portion.append(var_interval_time)

            #var_mix_3 = var_mix - var_space_interval - var_space_time - var_interval_time
            var_mix_3 = 1 - var_space - var_time - var_interval - var_space_interval - var_space_time - var_interval_time
            mix_3_portion.append(var_mix_3)

        time_portion = np.mean(np.array(time_portion))
        interval_portion = np.mean(np.array(interval_portion))
        space_portion = np.mean(np.array(space_portion))
        mix_portion = np.mean(np.array(mix_portion))
        time_interval_portion = np.mean(np.array(time_interval_portion))
        interval_space_portion = np.mean(np.array(interval_space_portion))
        time_space_portion = np.mean(np.array(time_space_portion))
        mix_3_portion = np.mean(np.array(mix_3_portion))

        print(time_portion, interval_portion, space_portion, time_interval_portion, interval_space_portion, time_space_portion, mix_3_portion)
    else:
        time_portion = 0.22114655214495782
        interval_portion = 0.026819198632679966
        space_portion = 0.6189156284786677
        time_interval_portion = 0.014553555019799451
        interval_space_portion = 0.012006758290429121
        time_space_portion = 0.09836356436756359
        mix_3_portion = 0.008194743065902366


    ########################
    # plot
    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.4*(2/2.5), 0.1, 0.4*(2/2.5), 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    barWidth = 0.5

    r0 = 0
    #names = ['SR', 't-SR', 'DM', 't-DM']
    plt.bar(r0, space_portion, color='#FFB300', edgecolor='white', width=barWidth*0.92,
            label="S"+": "+str(round(100*space_portion, 1))+"%")
    plt.bar(r0, time_portion, bottom=space_portion, color='#803E75', edgecolor='white', width=barWidth*0.92,
            label="F"+": "+str(round(100*time_portion, 1))+"%")
    plt.bar(r0, time_space_portion, bottom=time_portion+space_portion, color='#FF6800', edgecolor='white',
            width=barWidth*0.92, label="S+F"+": "+str(round(100*time_space_portion, 1))+"%")
    #plt.bar(r0, 1-space_portion-time_portion-time_space_portion, bottom=time_portion + space_portion+time_space_portion, color='black', alpha=0.5, edgecolor='white',
    #        width=barWidth * 0.92, label="Other" + ": " + str(round(100 * (1-space_portion-time_portion-time_space_portion), 1)) + "%")
    plt.bar(r0, 1 - space_portion - time_portion - time_space_portion,
            bottom=time_portion + space_portion + time_space_portion, color='white', edgecolor='black',linestyle="--",
            width=barWidth * 0.92, label='a')
    other_portion = 1-space_portion-time_portion-time_space_portion
    #################################################################
    r1 = 1
    plt.bar(r1, interval_portion/other_portion, color='#A6BDD7', edgecolor='white', width=barWidth*0.92,
            label="I"+": "+str(round(100*interval_portion, 1))+"%")
    plt.bar(r1, time_interval_portion/other_portion, bottom=interval_portion/other_portion, color='#C10020', edgecolor='white', width=barWidth*0.92,
            label="I+F"+": "+str(round(100*time_interval_portion, 1))+"%")
    plt.bar(r1, interval_space_portion/other_portion, bottom=(interval_portion+time_interval_portion)/other_portion, color='#CEA262', edgecolor='white',
            width=barWidth*0.92, label="I+S"+": "+str(round(100*interval_space_portion, 1))+"%")
    plt.bar(r1, mix_3_portion/other_portion, bottom=(interval_portion + time_interval_portion+interval_space_portion)/other_portion, color='#007D34', edgecolor='white',
            width=barWidth * 0.92, label="S+I+F" + ": " + str(round(100 * mix_3_portion, 1)) + "%")

    plt.ylabel('Portion of\nvariance', fontsize=fs)
    plt.plot([r0+barWidth*0.92/2, r1-barWidth*0.92/2], [1, 1], color='black', linewidth=1,linestyle="--")
    plt.plot([r0+barWidth*0.92/2, r1-barWidth*0.92/2], [time_portion + space_portion + time_space_portion, 0], color='black', linewidth=1,linestyle="--")

    # Custom x axis
    plt.xticks([], [])
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    #red_dot, = plt.plot(z, "ro", markersize=15)
    plt.legend(loc=(-1., 0.95), fontsize=fs, frameon=False, ncol=2, handlelength=0.7, columnspacing=1)

    # Show graphic
    plt.show()

    return fig


