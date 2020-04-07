import numpy as np
from matplotlib import pyplot as plt

import os

from numba import jit


from .. import run

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

plt.style.use('default')

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
        self.firing_rate_binder = firing_rate_binder#.astype(np.float64)
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


def weight_connection_with_time(serial_idxes, epoch='interval',  gamma_bar=np.array([1]), c=np.array([0.01, -0.01]), time_points=np.array([600]), func_relevance_threshold=2, noise_on=False):

    weight_connection_obj = WeightConnection(len(c))

    time_points = (time_points/20).astype(np.int)

    dly_interval = 1600
    prod_intervals = np.array([1200])

    prod_intervals, c = np.meshgrid(prod_intervals, c)
    prod_intervals = prod_intervals.flatten()
    c = c.flatten()

    batch_size = len(prod_intervals)
    gamma_bar = np.array([gamma_bar] * batch_size).flatten()

    dly_intervals = np.array([dly_interval] * batch_size)

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_decision_making/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_decision_making', is_cuda=False, noise_on=noise_on)

        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, dly_interval=dly_intervals,
                                                gamma_bar=gamma_bar, c=c)

        stim1_off, stim2_on = trial_input.epochs[epoch]

        firing_rate_binder = run_result.firing_rate_binder.detach().cpu().numpy()

        firing_rate_binder = np.concatenate(list(firing_rate_binder[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)), axis=0)

        weight_connection_obj.reset_parameter(firing_rate_binder, runnerObj.model.weight_hh.detach().cpu().numpy(), time_points)
        weight_connection_obj.do()

    p_mean, n_mean = weight_connection_obj.return_mean()
    p_std, n_std = weight_connection_obj.return_sem()

    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.3, 0.2, 0.7*0.9, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mean_same = np.concatenate([n_mean[0], p_mean[0]])
    std_same = np.concatenate([n_std[0], p_std[0]])
    x_coord_same = np.concatenate([-np.flip(np.arange(len(n_mean[0])), axis=0)-1, np.arange(len(p_mean[0]))])

    plt.plot(x_coord_same, mean_same, label='Same choice pref.')
    plt.fill_between(x_coord_same, mean_same-std_same, mean_same+std_same, alpha=0.5)

    mean_diff = np.concatenate([n_mean[1], p_mean[1]])
    std_diff = np.concatenate([n_std[1], p_std[1]])
    x_coord_diff = np.concatenate([-np.flip(np.arange(len(n_mean[1])), axis=0)-1, np.arange(len(p_mean[1]))])

    plt.plot(x_coord_diff, mean_diff, label='Diff. choice pref.')
    plt.fill_between(x_coord_diff, mean_diff-std_diff, mean_diff+std_diff, alpha=0.5)

    plt.gca().set_xlabel('Peak order difference', fontsize=fs)
    plt.gca().set_ylabel('Recurrent weight', fontsize=fs)

    #plt.plot([-13, 11], [0, 0], '--', color='black')
    min_x = np.min([x_coord_same[0], x_coord_diff[0]])
    max_x = np.max([x_coord_same[-1], x_coord_diff[-1]])

    plt.plot([min_x, max_x], [0, 0], '--', color='black')
    plt.xlim([min_x, max_x])

    if min_x<=-10 and max_x>=10:
        plt.xticks([-10,0,9], [-10, 1, 10], fontsize=fs)
    else:
        plt.xticks([-10+5,0,9-5], [-10+5, 1, 10-5], fontsize=fs)

    fig.legend(loc='best', fontsize=fs, frameon=False)

    plt.show()
    return fig
