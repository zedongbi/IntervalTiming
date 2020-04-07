from __future__ import division

import json

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm

import os
from scipy import stats
from collections import defaultdict

from .. import run

from .. import tools

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

plt.style.use('default')

#data_pca shape: (stimulus, time, neuron)
def data_rescale_fcn(data_pca, maxT=None):

    # rescale time of data_pca
    if maxT is None:
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


def correlation_decimation(angle, mix_var, decim_num=5):
    delete_idx_list = list()
    # angle_var_0 = stats.pearsonr(np.array(angle), np.array(mix_var))[0]

    for k in range(decim_num):
        smallest_corr = 1000.
        smallest_idx = -1
        for i in range(len(angle)):
            angle_temp = np.delete(angle, i)
            mix_var_temp = np.delete(mix_var, i)
            corr = np.abs((stats.pearsonr(np.array(angle_temp), np.array(mix_var_temp)))[0])
            if corr < smallest_corr:
                smallest_corr = corr
                smallest_idx = i
        angle = np.delete(angle, smallest_idx)
        mix_var = np.delete(mix_var, smallest_idx)
        delete_idx_list.append(smallest_idx)

    for i in range(len(delete_idx_list) - 2, -1, -1):
        for j in range(i + 1, len(delete_idx_list)):
            if delete_idx_list[j] >= delete_idx_list[i]:
                delete_idx_list[j] = delete_idx_list[j] + 1
    return delete_idx_list

# if after removing a single point, the correlation value is changed by more than 0.3,
# then the correlation is considered to be very sensitive to this single point
# we remove this single point for a more robust result
def correlation_exclude(angle, mix_var, decoding):
    angle_decoding = (stats.pearsonr(np.array(angle), np.array(decoding)))[0]
    var_decoding = (stats.pearsonr(np.array(mix_var), np.array(decoding)))[0]

    exclude_list = list()
    for i in range(len(decoding)):
        angle_temp = np.delete(angle, i)
        mix_var_temp = np.delete(mix_var, i)
        decoding_temp = np.delete(decoding, i)

        angle_decoding_temp = (stats.pearsonr(np.array(angle_temp), np.array(decoding_temp)))[0]
        var_decoding_temp = (stats.pearsonr(np.array(mix_var_temp), np.array(decoding_temp)))[0]

        if abs(angle_decoding - angle_decoding_temp) > 0.3:
            exclude_list.append(i)

        if abs(var_decoding - var_decoding_temp) > 0.3:
            exclude_list.append(i)

    return exclude_list


def time_flow_decoding_generalization_across_space(serial_idxes, epoch='go', decoder_type=1, transient_period=10, noise_on=False, fig_on=True, go_scaling=False, redo=False):
    '''
    Decoding generalization of temporal flow across space, as a function of angle and mixed variance
    '''
    #def get_dist(original_dist):
    #    '''Get the distance in periodic boundary conditions'''
    #    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    result_dict = dict()

    tools.mkdir_p("space_decoding_data/")

    fname = os.path.join('./space_decoding_data/', 'time_flow_decoding_generalization_across_space.json')
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            result_dict = json.load(f)

    filekey = 'epoch_'+epoch+'decoder_type'+str(decoder_type)+'transient_'+str(transient_period)+'go_scaling'+str(go_scaling)

    if filekey in result_dict and redo == False:
        total_result_decoding, total_result_mix_var, total_result_angle = result_dict[filekey]
    else:
        interval_for_test = np.array([12-1, 18-1, 24-1, 30-1, 36-1, 42-1, 48-1, 54-1, 60-1])
        interval_for_test = interval_for_test - transient_period
        interval_for_test = interval_for_test[interval_for_test>=0]

        prod_interval = 1200

        total_result_decoding = list()
        total_result_mix_var = list()
        total_result_angle = list()

        for serial_idx in serial_idxes:

            result = defaultdict(list)

            model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

            if not os.path.exists(model_dir):
                continue

            ring_centers = np.arange(6., 32.-6)
            batch_size = len(ring_centers)
            prod_intervals = np.array([prod_interval] * batch_size)
            dly_intervals = np.array([1600] * batch_size)

            runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=noise_on)
            trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                    dly_interval=dly_intervals, gaussian_center=ring_centers)

            firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
            ##########################################################################################

            if epoch == 'interval':
                stim1_off, stim2_on = trial_input.epochs[epoch]
                stim1_off = stim1_off + transient_period

                # (stimulus, time, neuron)
                firing_rate_list_for_all = np.concatenate(
                    list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)),
                    axis=0)
                ################################################################
                # shape: neuron, time, batch_for_direction,
                firing_rate_list = firing_rate_list_for_all.transpose((2, 1, 0))

                firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
                firing_rate_list_centered = firing_rate_list - firing_rate_list_mean

                firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
                var_t = np.var(firing_rate_list_t) / np.var(firing_rate_list_centered)

                firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)
                var_s = np.var(firing_rate_list_s) / np.var(firing_rate_list_centered)

                total_result_mix_var.append(1-var_t-var_s)

                ##############################################
                firing_rate_list = firing_rate_list_for_all
                # component of time
                concate_firing_rate_time = np.mean(firing_rate_list, axis=0)
                pca_time = PCA(n_components=2)
                pca_time.fit(concate_firing_rate_time)

                # component of stimulus
                concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)
                pca_stim = PCA(n_components=2)
                pca_stim.fit(concate_firing_rate_stim)

                total_result_angle.append(180/np.pi*np.arccos(np.abs(np.sum(pca_time.components_[0] * pca_stim.components_[0]))))

                ########################################
                firing_rate_list = firing_rate_list_for_all
                firing_rate_list = firing_rate_list.transpose((1, 0, 2))

                for ref_idx in range(batch_size):
                    #print('origin', ref_idx)
                    dir_ref = ring_centers[ref_idx]
                    firing_rate_ref = firing_rate_list[:, ref_idx, :]

                    pca_time = PCA(n_components=1)
                    pca_time.fit(firing_rate_ref)
                    firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                    transform_matrix = pca_time.components_.T


                    for aux_idx in range(batch_size):
                        dir_aux = ring_centers[aux_idx]
                        firing_rate_aux = firing_rate_list[:, aux_idx, :]

                        #firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0), transform_matrix)
                        if decoder_type==1:
                            firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_ref, axis=0), transform_matrix)
                        else:
                            firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0), transform_matrix)

                        for time_ptr in interval_for_test:
                            state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                            distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux)**2, axis=1))
                            min_time_ptr = np.argmin(distance_with_ref)
                            result[round(abs(dir_ref-dir_aux), 6)].append(abs(min_time_ptr-time_ptr)*20)

                temp_result = list()
                for k, v in result.items():
                    mean_v = np.mean(np.array(v))
                    temp_result.append(mean_v)
                total_result_decoding.append(np.mean(np.array(temp_result)))
            ##########################################################################################

            if epoch == 'go':

                if go_scaling == False:

                    stim1_off, stim2_on = trial_input.epochs[epoch]
                    stim1_off = stim1_off + transient_period

                    # (stimulus, time, neuron)
                    firing_rate_list_for_all = np.concatenate(
                        list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :, :] for i in range(0, batch_size)),
                        axis=0)

                else:
                    stim1_off, stim2_on = trial_input.epochs[epoch]

                    outputs = np.max(run_result.outputs.detach().cpu().numpy(), axis=2)
                    motion_time = np.array([np.argmax(outputs[stim1_off[i]:, i] > 0.5) for i in range(batch_size)])

                    stim2_on = stim1_off + motion_time
                    firing_rate_list_for_all = list(firing_rate[stim1_off[i]:stim2_on[i], i, :] for i in range(0, batch_size))
                    firing_rate_list_for_all = data_rescale_fcn(firing_rate_list_for_all, int(prod_interval/20))
                    firing_rate_list_for_all = firing_rate_list_for_all[:, transient_period:, :]

                #################################################################################
                # shape: neuron, time, batch_for_direction,
                firing_rate_list = firing_rate_list_for_all.transpose((2, 1, 0))

                firing_rate_list_mean = np.mean(firing_rate_list, axis=(1, 2))[:, np.newaxis, np.newaxis]
                firing_rate_list_centered = firing_rate_list - firing_rate_list_mean

                firing_rate_list_t = np.mean(firing_rate_list_centered, axis=2)
                var_t = np.var(firing_rate_list_t) / np.var(firing_rate_list_centered)

                firing_rate_list_s = np.mean(firing_rate_list_centered, axis=1)
                var_s = np.var(firing_rate_list_s) / np.var(firing_rate_list_centered)

                total_result_mix_var.append(1-var_t-var_s)
                ##############################################

                firing_rate_list = firing_rate_list_for_all
                # component of time
                concate_firing_rate_time = np.mean(firing_rate_list, axis=0)
                pca_time = PCA(n_components=2)
                pca_time.fit(concate_firing_rate_time)

                # component of stimulus
                concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)
                pca_stim = PCA(n_components=2)
                pca_stim.fit(concate_firing_rate_stim)

                total_result_angle.append(180/np.pi*np.arccos(np.abs(np.sum(pca_time.components_[0] * pca_stim.components_[0]))))

                ########################################

                firing_rate_list = firing_rate_list_for_all
                firing_rate_list = firing_rate_list.transpose((1, 0, 2))

                for ref_idx in range(batch_size):
                    # print('origin', ref_idx)
                    dir_ref = ring_centers[ref_idx]
                    firing_rate_ref = firing_rate_list[:, ref_idx, :]

                    pca_time = PCA(n_components=1)
                    pca_time.fit(firing_rate_ref)
                    firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                    transform_matrix = pca_time.components_.T

                    for aux_idx in range(batch_size):
                        dir_aux = ring_centers[aux_idx]
                        firing_rate_aux = firing_rate_list[:, aux_idx, :]

                        #firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0), transform_matrix)
                        if decoder_type==1:
                            firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_ref, axis=0), transform_matrix)
                        else:
                            firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0), transform_matrix)

                        for time_ptr in interval_for_test:
                            state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                            distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux) ** 2, axis=1))
                            min_time_ptr = np.argmin(distance_with_ref)
                            result[round(abs(dir_ref - dir_aux), 6)].append(abs(min_time_ptr - time_ptr) * 20)

                temp_result = list()
                for k, v in result.items():
                    mean_v = np.mean(np.array(v))
                    temp_result.append(mean_v)
                total_result_decoding.append(np.mean(np.array(temp_result)))
        ###########################################################################
        result_dict[filekey] = (total_result_decoding, total_result_mix_var, total_result_angle)
        with open(fname, 'w') as f:
            json.dump(result_dict, f)
    #print('total_result_decoding: ', total_result_decoding)
    #print('total_result_mix_var: ', total_result_mix_var)
    #print('total_result_angle: ', total_result_angle)

    if fig_on:
        fig = plt.figure(figsize=(2.5, 2.1))
        ax = fig.add_axes([0.2, 0.3, 0.5, 0.5])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.text(0.18, 72, '$\\rho$=-0.3')
        #ax.text(0., 95, '$\\rho$=-0.3')

        max_decoding_error = max(total_result_decoding)
        min_decoding_error = min(total_result_decoding)
        m = cm.ScalarMappable(cmap='rainbow')
        m.set_clim(vmin=min_decoding_error, vmax=max_decoding_error)

        is_color=True
        if is_color:
            color = [m.to_rgba(x) for x in total_result_decoding]
        else:
            color = 'black'
        plt.scatter(total_result_mix_var, total_result_angle, c=color, marker='o')

        #for i, txt in enumerate(total_result_mix_var):
        #    ax.annotate(i, (total_result_mix_var[i], total_result_angle[i]))

        plt.xticks(fontsize=fs)
        plt.yticks([60, 70, 80, 90], fontsize=fs)
        plt.gca().set_xlabel('Portion of\nmixed variance', fontsize=fs)
        plt.gca().set_ylabel('Angle (deg)', fontsize=fs)

        if is_color:
            # color bar
            cmap = plt.get_cmap('rainbow')#mpl.rcParams["image.cmap"]
            m = cm.ScalarMappable(cmap=cmap)
            m.set_array([0, 1])
            m.set_clim(vmin=min_decoding_error, vmax=max_decoding_error)

            cbar = fig.colorbar(m)
            #cbar.set_ticks([0, 0.5, 1])
            cbar.ax.tick_params(labelsize=fs)
            cbar.ax.set_title('Decoding\n error (ms)', fontsize=fs)

        #print(stats.pearsonr(np.array(total_result_angle), np.array(total_result_decoding)))
        #print(stats.pearsonr(np.array(total_result_mix_var), np.array(total_result_decoding)))
        #print(stats.pearsonr(np.array(total_result_angle), np.array(total_result_mix_var)))

        plt.show()
        return fig
    return total_result_decoding, total_result_mix_var, total_result_angle


def direct_correlation_by_decimation(angle, mix_var, decoding, decim_num=5):
    angle_var_0 = stats.pearsonr(np.array(angle), np.array(mix_var))[0]
    angle_decoding_0 = stats.pearsonr(np.array(angle), np.array(decoding))[0]
    var_decoding_0 = stats.pearsonr(np.array(mix_var), np.array(decoding))[0]

    for k in range(decim_num):
        smallest_corr = 1000.
        smallest_idx = -1
        for i in range(len(decoding)):
            angle_temp = np.delete(angle, i)
            mix_var_temp = np.delete(mix_var, i)
            corr = np.abs((stats.pearsonr(np.array(angle_temp), np.array(mix_var_temp)))[0])
            if corr < smallest_corr:
                smallest_corr = corr
                smallest_idx = i
        angle = np.delete(angle, smallest_idx)
        mix_var = np.delete(mix_var, smallest_idx)
        decoding = np.delete(decoding, smallest_idx)

    angle_var_1 = stats.pearsonr(np.array(angle), np.array(mix_var))[0]
    angle_decoding_1 = stats.pearsonr(np.array(angle), np.array(decoding))[0]
    var_decoding_1 = stats.pearsonr(np.array(mix_var), np.array(decoding))[0]


    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.3, 0.3, 0.2, 0.5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)

    x_pos = [0, 0.8]
    performance = [angle_decoding_1, var_decoding_1]
    barWidth = 0.75
    ax.bar(x_pos, performance, align='center', edgecolor='black', linewidth=2, color='#A6BDD7')

    ax.set_xticks([0-barWidth, 0.8-barWidth/3])
    ax.set_xticklabels(['DE-AG', 'DE-MV'], rotation=45)

    ax.tick_params('x', length=0)
    xmax = ax.get_xlim()[1]

    ax.set_ylabel('Correlation', fontsize=fs)

    ax.plot([ax.get_xlim()[1], ax.get_xlim()[0]], [0, 0], 'black', linewidth=1)

    plt.show()
    return fig


def time_flow_generalization(serial_idxes, epoch='interval', noise_on=False):
    '''
    plot perception and production epoch together
    '''
    # def get_dist(original_dist):
    #    '''Get the distance in periodic boundary conditions'''
    #    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    if epoch == 'interval':
        transient_period = 20
        interval_for_test = np.array([24 - 1, 30 - 1, 36 - 1, 42 - 1, 48 - 1, 54 - 1, 60 - 1])
    else:
        transient_period = 10
        interval_for_test = np.array([12 - 1, 18 - 1, 24 - 1, 30 - 1, 36 - 1, 42 - 1, 48 - 1, 54 - 1, 60 - 1])

    interval_for_test = interval_for_test - transient_period
    prod_interval = 1200

    total_result_decoding_type_1 = defaultdict(list)
    total_result_decoding_type_2 = defaultdict(list)

    for serial_idx in serial_idxes:

        model_dir = './core/model/' + 'timed_spatial_reproduction/' + str(serial_idx)

        if not os.path.exists(model_dir):
            continue

        ring_centers = np.arange(6., 32. - 6, 2.)
        batch_size = len(ring_centers)
        prod_intervals = np.array([prod_interval] * batch_size)
        dly_intervals = np.array([1600] * batch_size)

        runnerObj = run.Runner(model_dir=model_dir, rule_name='timed_spatial_reproduction', is_cuda=False, noise_on=noise_on)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals,
                                                dly_interval=dly_intervals, gaussian_center=ring_centers)

        firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
        ##########################################################################################

        if epoch == 'interval':
            stim1_off, stim2_on = trial_input.epochs[epoch]
            stim1_off = stim1_off + transient_period
            firing_rate_list = np.concatenate(
                list(firing_rate[stim1_off[i]:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
                axis=1).squeeze()

            result = defaultdict(list)

            for ref_idx in range(batch_size):
                # print('origin', ref_idx)
                dir_ref = ring_centers[ref_idx]
                firing_rate_ref = firing_rate_list[:, ref_idx, :]

                pca_time = PCA(n_components=1)
                pca_time.fit(firing_rate_ref)
                firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                transform_matrix = pca_time.components_.T

                for aux_idx in range(batch_size):
                    dir_aux = ring_centers[aux_idx]
                    firing_rate_aux = firing_rate_list[:, aux_idx, :]

                    # firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0), transform_matrix)
                    firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_ref, axis=0),
                                                     transform_matrix)

                    for time_ptr in interval_for_test:
                        state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                        distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux) ** 2, axis=1))
                        min_time_ptr = np.argmin(distance_with_ref)
                        result[round(abs(dir_ref - dir_aux), 6)].append(abs(min_time_ptr - time_ptr) * 20)

            for k, v in result.items():
                mean_v = np.mean(np.array(v))
                total_result_decoding_type_1[k].append(mean_v)
            #################################################
            result = defaultdict(list)

            for ref_idx in range(batch_size):
                # print('origin', ref_idx)
                dir_ref = ring_centers[ref_idx]
                firing_rate_ref = firing_rate_list[:, ref_idx, :]

                pca_time = PCA(n_components=1)
                pca_time.fit(firing_rate_ref)
                firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                transform_matrix = pca_time.components_.T

                for aux_idx in range(batch_size):
                    dir_aux = ring_centers[aux_idx]
                    firing_rate_aux = firing_rate_list[:, aux_idx, :]

                    firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0),
                                                     transform_matrix)
                    # firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_ref, axis=0), transform_matrix)

                    for time_ptr in interval_for_test:
                        state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                        distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux) ** 2, axis=1))
                        min_time_ptr = np.argmin(distance_with_ref)
                        result[round(abs(dir_ref - dir_aux), 6)].append(abs(min_time_ptr - time_ptr) * 20)

            for k, v in result.items():
                mean_v = np.mean(np.array(v))
                total_result_decoding_type_2[k].append(mean_v)
        ##########################################################################################

        if epoch == 'go':
            stim1_off, stim2_on = trial_input.epochs[epoch]
            stim1_off = stim1_off + transient_period

            firing_rate_list = np.concatenate(
                list(firing_rate[stim1_off[i]:stim2_on[i], i, :][:, np.newaxis, :] for i in range(0, batch_size)),
                axis=1).squeeze()

            result = defaultdict(list)

            for ref_idx in range(batch_size):
                # print('origin', ref_idx)
                dir_ref = ring_centers[ref_idx]
                firing_rate_ref = firing_rate_list[:, ref_idx, :]

                pca_time = PCA(n_components=1)
                pca_time.fit(firing_rate_ref)
                firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                transform_matrix = pca_time.components_.T

                for aux_idx in range(batch_size):
                    dir_aux = ring_centers[aux_idx]
                    firing_rate_aux = firing_rate_list[:, aux_idx, :]

                    firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_ref, axis=0),
                                                     transform_matrix)

                    for time_ptr in interval_for_test:
                        state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                        distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux) ** 2, axis=1))
                        min_time_ptr = np.argmin(distance_with_ref)
                        result[round(abs(dir_ref - dir_aux), 6)].append(abs(min_time_ptr - time_ptr) * 20)

            for k, v in result.items():
                mean_v = np.mean(np.array(v))
                total_result_decoding_type_1[k].append(mean_v)

            ####################
            result = defaultdict(list)

            for ref_idx in range(batch_size):
                # print('origin', ref_idx)
                dir_ref = ring_centers[ref_idx]
                firing_rate_ref = firing_rate_list[:, ref_idx, :]

                pca_time = PCA(n_components=1)
                pca_time.fit(firing_rate_ref)
                firing_rate_time_ref = pca_time.transform(firing_rate_ref)
                transform_matrix = pca_time.components_.T

                for aux_idx in range(batch_size):
                    dir_aux = ring_centers[aux_idx]
                    firing_rate_aux = firing_rate_list[:, aux_idx, :]

                    firing_rate_time_aux = np.matmul(firing_rate_aux - np.mean(firing_rate_aux, axis=0),
                                                     transform_matrix)

                    for time_ptr in interval_for_test:
                        state_of_aux = firing_rate_time_aux[time_ptr, :][np.newaxis, :]
                        distance_with_ref = np.sqrt(np.sum((firing_rate_time_ref - state_of_aux) ** 2, axis=1))
                        min_time_ptr = np.argmin(distance_with_ref)
                        result[round(abs(dir_ref - dir_aux), 6)].append(abs(min_time_ptr - time_ptr) * 20)

            for k, v in result.items():
                mean_v = np.mean(np.array(v))
                total_result_decoding_type_2[k].append(mean_v)
        ###################################################################

    dir_diff_type_1 = list()
    decode_error_mean_type_1 = list()
    decode_error_sem_type_1 = list()
    for k, v in total_result_decoding_type_1.items():
        dir_diff_type_1.append(k)
        decode_error_mean_type_1.append(np.mean(np.array(v)))
        decode_error_sem_type_1.append(np.std(np.array(v)) / np.sqrt(len(v)))

    dir_diff_type_2 = list()
    decode_error_mean_type_2 = list()
    decode_error_sem_type_2 = list()
    for k, v in total_result_decoding_type_2.items():
        dir_diff_type_2.append(k)
        decode_error_mean_type_2.append(np.mean(np.array(v)))
        decode_error_sem_type_2.append(np.std(np.array(v)) / np.sqrt(len(v)))

    '''
    dir_diff_shuffle = list()
    decode_error_shuffle = list()
    for k, v in total_result_shuffle.items():
        dir_diff_shuffle.append(k)
        decode_error_shuffle.append(np.mean(np.array(v)))
    '''
    # fig = plt.figure(figsize=(2.5, 2.1))
    # ax = fig.add_axes([0.35, 0.22, 0.65*0.8, 0.75*0.5])
    fig = plt.figure(figsize=(1.8, 2.1))
    ax = fig.add_axes([0.4, 0.29, 0.58, 0.69])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.errorbar(time_interval_mean, velocity_mean, xerr=time_interval_sem, yerr=velocity_sem)
    # ax.plot(dir_diff, decode_error)
    if epoch == 'interval':
        ax.errorbar(dir_diff_type_1, decode_error_mean_type_1, yerr=decode_error_sem_type_1, color='magenta')
        ax.errorbar(dir_diff_type_2, decode_error_mean_type_2, yerr=decode_error_sem_type_2, linestyle='--',
                    color='magenta')

    if epoch == 'go':
        ax.errorbar(dir_diff_type_1, decode_error_mean_type_1, yerr=decode_error_sem_type_1, color='green')
        ax.errorbar(dir_diff_type_2, decode_error_mean_type_2, yerr=decode_error_sem_type_2, linestyle='--',
                    color='green')

    # ax.plot(dir_diff_interval, np.array([450]*len(dir_diff_interval)), '--', color='black')
    if epoch == 'go':
        ax.plot(dir_diff_type_1, np.array([375] * len(dir_diff_type_1)), '--', color='black')
    else:
        ax.plot(dir_diff_type_1, np.array([300] * len(dir_diff_type_1)), '--', color='black')

    # ax.set_xlabel('Spatial separation', fontsize=fs)
    ax.set_xlabel('Spatial\nseparation', fontsize=fs)
    ax.set_ylabel('Decoding error (ms)', fontsize=fs)

    # plt.xticks([0, np.pi/2, np.pi], [0, '$\pi/2$', '$\pi$'], fontsize=fs)

    plt.show()
    return fig
