from matplotlib import pyplot as plt
import numpy as np
import errno

from scipy import stats
import json
import pickle
import os


from core.data_plot import nontiming_analysis_1
from core.data_plot import nontiming_analysis_2
from core_multitasking.data_plot import nontiming_analysis as nontiming_analysis_multitasking
from core_feedback.data_plot import nontiming_analysis as nontiming_analysis_feedback

dot_alpha=0.4

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

mkdir_p("./figure/fig6")

fs = 10 # font size
plt.style.use('default')

# sky blue, green, pink, kelly-vivid-yellow, kelly-vivid-red
color_list = ['#75bbfd', '#15b01a', '#ff81c0', '#FFB300', '#C10020']


def portion_of_variance(serial_idxes):
    spatial_reproduction_t, spatial_reproduction_s = nontiming_analysis_2.temporal_and_stim_variance_spatial_reproduction(serial_idxes)
    spatial_reproduction_t = np.array(spatial_reproduction_t)
    spatial_reproduction_s = np.array(spatial_reproduction_s)

    timed_spatial_reproduction_t, timed_spatial_reproduction_s = nontiming_analysis_2.temporal_and_stim_variance_timed_spatial_reproduction(serial_idxes)
    timed_spatial_reproduction_t = np.array(timed_spatial_reproduction_t)
    timed_spatial_reproduction_s = np.array(timed_spatial_reproduction_s)

    decision_making_t, decision_making_d = nontiming_analysis_1.temporal_spatial_variance_decision_making(serial_idxes)#nontiming_analysis_1.temporal_and_decision_variance_decision_making(serial_idxes)
    decision_making_t = np.array(decision_making_t)
    decision_making_d = np.array(decision_making_d)

    timed_decision_making_t, timed_decision_making_d = nontiming_analysis_1.temporal_spatial_variance_timed_decision_making(serial_idxes)#nontiming_analysis_1.temporal_and_decision_variance_timed_decision_making(serial_idxes)
    timed_decision_making_t = np.array(timed_decision_making_t)
    timed_decision_making_d = np.array(timed_decision_making_d)

    #print(np.mean(spatial_reproduction_t), np.mean(spatial_reproduction_s))
    #print(np.mean(timed_spatial_reproduction_t), np.mean(timed_spatial_reproduction_s))
    #print(np.mean(decision_making_t), np.mean(decision_making_d))
    #print(np.mean(timed_decision_making_t), np.mean(timed_decision_making_d))

    spatial_reproduction_t = np.mean(spatial_reproduction_t)
    spatial_reproduction_s = np.mean(spatial_reproduction_s)
    spatial_reproduction_ts = 1 - spatial_reproduction_t - spatial_reproduction_s

    timed_spatial_reproduction_t = np.mean(timed_spatial_reproduction_t)
    timed_spatial_reproduction_s = np.mean(timed_spatial_reproduction_s)
    timed_spatial_reproduction_ts = 1 - timed_spatial_reproduction_t - timed_spatial_reproduction_s

    decision_making_t = np.mean(decision_making_t)
    decision_making_d = np.mean(decision_making_d)
    decision_making_td = 1 - decision_making_t - decision_making_d

    timed_decision_making_t = np.mean(timed_decision_making_t)
    timed_decision_making_d = np.mean(timed_decision_making_d)
    timed_decision_making_td = 1 - timed_decision_making_t - timed_decision_making_d

    t_variance = [spatial_reproduction_t, timed_spatial_reproduction_t, decision_making_t, timed_decision_making_t]
    s_variance = [spatial_reproduction_s, timed_spatial_reproduction_s, decision_making_d, timed_decision_making_d]
    ts_variance = [spatial_reproduction_ts, timed_spatial_reproduction_ts, decision_making_td, timed_decision_making_td]

    # plot
    fig = plt.figure(figsize=(2.5, 2.1))
    ax = fig.add_axes([0.3, 0.2, 0.57, 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    barWidth = 0.5
    r = [0, barWidth, 2*barWidth, 3*barWidth]
    names = ['SR', 't-SR', 'DM', 't-DM']
    # Create green Bars
    plt.bar(r, t_variance, color='#b5ffb9', edgecolor='white', width=barWidth*0.92, label="Time")
    # Create orange Bars
    plt.bar(r, s_variance, bottom=t_variance, color='#f9bc86', edgecolor='white', width=barWidth*0.92, label="Space or decision")
    # Create blue Bars
    plt.bar(r, ts_variance, bottom=[i + j for i, j in zip(t_variance, s_variance)], color='#a3acff', edgecolor='white',
            width=barWidth*0.92, label="Mixed")
    plt.ylabel('Portion of variance', fontsize=fs)

    # Custom x axis
    plt.xticks(r, names)
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.legend(loc=(-0.2, 0.95), fontsize=fs, frameon=False, ncol=1)

    # Show graphic
    plt.show()

    return fig


fig = portion_of_variance(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(15)])
fig.savefig("./figure/fig6/portion_of_variance.pdf")


def temporal_variance_by_tuning_broadness(serial_idxes):
    frequency_prod_dly_t, frequency_prod_dly_broad_tuning_t, \
    spatial_reproduction_t, spatial_reproduction_broad_tuning_t, \
    spatial_comparison_t, spatial_comparison_broad_tuning_t, \
    spatial_change_detection_t, spatial_change_detection_broad_tuning_t = nontiming_analysis_2.temporal_variance(serial_idxes)

    #print(stats.ttest_ind(frequency_prod_dly_t, frequency_prod_dly_broad_tuning_t, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_reproduction_t, spatial_reproduction_broad_tuning_t, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_comparison_t, spatial_comparison_broad_tuning_t, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_change_detection_t, spatial_change_detection_broad_tuning_t, equal_var=False)[1])
    #############################################################################################
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.68, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    width = 0.5
    dots_random_width = 0.05
    single_result = np.array(frequency_prod_dly_t)
    counter = 0
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10, label='unbroadened tuning')

    single_result = frequency_prod_dly_broad_tuning_t
    counter = 0
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10, label='broadened tuning')

    if stats.ttest_ind(frequency_prod_dly_t, frequency_prod_dly_broad_tuning_t, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_reproduction_t
    counter = 1.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_reproduction_broad_tuning_t
    counter = 1.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_reproduction_t, spatial_reproduction_broad_tuning_t, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_comparison_t
    counter = 3
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_comparison_broad_tuning_t
    counter = 3
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_comparison_t, spatial_comparison_broad_tuning_t, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_change_detection_t
    counter = 4.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_change_detection_broad_tuning_t
    counter = 4.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_change_detection_t, spatial_change_detection_broad_tuning_t, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    #plt.xlabel('Peak time diff. (ms)', fontsize=fs)
    plt.ylabel('Temporal Portion\nof Variance', fontsize=fs)
    plt.xticks([1,1+1.5,1+3,1+4.5], ['t-SR', 'SR', 'COMP', 'CD'], fontsize=fs)

    #plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_ylim([0, 0.9])
    plt.text(0, 1.05, 'Control', color=color_list[0], fontsize=fs)
    plt.text(0, 0.95, 'Broad tuning', color=color_list[1], fontsize=fs)

    #plt.legend(loc=(0.1, 0.6), fontsize=fs, frameon=False)
    #fig.legend(loc='upper right', fontsize=fs, frameon=False)

    plt.show()
    return fig


fig=temporal_variance_by_tuning_broadness(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/temporal_variance_by_tuning_broadness.pdf")


def temporal_variance_by_task_complexity(serial_idxes):
    spatial_reproduction = nontiming_analysis_2.temporal_variance_spatial_reproduction(serial_idxes)
    spatial_change_detection = nontiming_analysis_2.temporal_variance_spatial_change_detection(serial_idxes)
    spatial_comparison = nontiming_analysis_2.temporal_variance_spatial_comparison(serial_idxes)

    simple_decision_making = nontiming_analysis_1.temporal_variance_decision_making(serial_idxes)
    ctx_decision_making = nontiming_analysis_1.temporal_variance_ctx_decision_making(serial_idxes)

    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    width = 0.5
    dots_random_width = 0.05
    single_result = np.array(spatial_reproduction)
    counter = 0
    x = np.random.normal(1+counter, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_change_detection
    counter = 0.7
    x = np.random.normal(1+counter, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_comparison
    counter = 0.7 * 2
    x = np.random.normal(1+counter, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    if stats.ttest_ind(spatial_reproduction, spatial_comparison, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1 + 0.35, 0.92), xytext=(1 + 0.35, 0.94), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=0.4', lw=1.0))

    plt.plot([1+counter+0.5, 1+counter+0.5], [0, 0.9], '--', color='black', lw=2)

    single_result = simple_decision_making
    counter = 0.7 * 2 + 1
    x = np.random.normal(1+counter, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = ctx_decision_making
    counter = 0.7 * 2 + 1 + 0.7
    x = np.random.normal(1+counter, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    if stats.ttest_ind(simple_decision_making, ctx_decision_making, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1 + 0.7 * 2 + 1 + 0.35, 0.92), xytext=(1+0.7 * 2 + 1 +0.35, 0.94), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=0.4', lw=1.0))

    #plt.xlabel('Peak time diff. (ms)', fontsize=fs)
    plt.ylabel('Temporal Portion\nof Variance', fontsize=fs)
    plt.xticks([1,1+0.7,1+0.7*2,1+0.7*2+1,1+0.7*2+1+0.7], ['SR', 'CD', 'COMP', 'DM', 'cue-DM'], fontsize=fs)

    #plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_ylim([0, 0.92])

    plt.show()
    return fig


fig=temporal_variance_by_task_complexity(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/temporal_variance_by_task_complexity.pdf")


def temporal_variance_by_multitask(serial_idxes):
    single_task_spatial_reproduction = nontiming_analysis_2.temporal_variance_spatial_reproduction(serial_idxes)
    multi_task_spatial_reproduction = nontiming_analysis_multitasking.temporal_variance_spatial_reproduction(serial_idxes)
    #single_task_broad_tuning = temporal_analysis_19.temporal_variance_spatial_reproduction_broad_tuning(serial_idxes)
    #multi_task_broad_tuning = temporal_analysis_20.temporal_variance_spatial_reproduction_broad_tuning(serial_idxes)

    single_task_decision_making = nontiming_analysis_1.temporal_variance_decision_making(serial_idxes)
    multi_task_decision_making = nontiming_analysis_multitasking.temporal_variance_decision_making(serial_idxes)

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.68, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    width = 0.5
    dots_random_width = 0.05
    single_result = np.array(single_task_spatial_reproduction)
    counter = 0
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = multi_task_spatial_reproduction
    counter = 0
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(single_task_spatial_reproduction, multi_task_spatial_reproduction, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.95), xytext=(1+counter, 0.97), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=0.4', lw=1.0))

    single_result = single_task_decision_making
    counter = 1.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = multi_task_decision_making
    counter = 1.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(single_task_decision_making, multi_task_decision_making, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.95), xytext=(1+counter, 0.97), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=0.4', lw=1.0))

    #plt.xlabel('Peak time diff. (ms)', fontsize=fs)
    plt.ylabel('Temporal Portion\nof Variance', fontsize=fs)
    plt.xticks([1-width/2, 1+width/2, 1+1.5-width/2, 1+1.5+width/2], ['SR', 'SR with\nt-SR', 'DM', 'DM with\nt-DM'], fontsize=fs, rotation=20)

    #plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_ylim([0, 0.95])
    #plt.text(0, 1.05, 'single task', color=color_list[0], fontsize=fs)
    #plt.text(0, 0.95, 'multi-task', color=color_list[1], fontsize=fs)

    #plt.legend(loc=(0.1, 0.6), fontsize=fs, frameon=False)
    #fig.legend(loc='upper right', fontsize=fs, frameon=False)

    plt.show()
    return fig


fig=temporal_variance_by_multitask(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/temporal_variance_by_multitask.pdf")


def temporal_variance_by_feedback(serial_idxes, serial_idxes_feedback):
    spatial_reproduction = nontiming_analysis_2.temporal_variance_spatial_reproduction(serial_idxes)
    spatial_reproduction_feedback = nontiming_analysis_feedback.temporal_variance_spatial_reproduction(serial_idxes_feedback)

    spatial_comparison = nontiming_analysis_2.temporal_variance_spatial_comparison(serial_idxes)
    spatial_comparison_timing_feedback = nontiming_analysis_feedback.temporal_variance_spatial_comparison(serial_idxes_feedback)

    spatial_change_detection = nontiming_analysis_2.temporal_variance_spatial_change_detection(serial_idxes)
    spatial_change_detection_timing_feedback = nontiming_analysis_feedback.temporal_variance_spatial_change_detection(serial_idxes_feedback)

    ctx_decision_making = nontiming_analysis_1.temporal_variance_ctx_decision_making(serial_idxes)
    ctx_decision_making_feedback = nontiming_analysis_feedback.temporal_variance_ctx_decision_making(serial_idxes_feedback)

    #print(stats.ttest_ind(spatial_reproduction, spatial_reproduction_feedback, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_comparison, spatial_comparison_timing_feedback, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_change_detection, spatial_change_detection_timing_feedback, equal_var=False)[1])
    #print(stats.ttest_ind(ctx_decision_making, ctx_decision_making_feedback, equal_var=False)[1])

    #############################################################################################
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.68, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    width = 0.5
    dots_random_width = 0.05
    single_result = np.array(spatial_reproduction)
    counter = 0
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_reproduction_feedback
    counter = 0
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_reproduction, spatial_reproduction_feedback, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_comparison
    counter = 1.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_comparison_timing_feedback
    counter = 1.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_comparison, spatial_comparison_timing_feedback, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_change_detection
    counter = 3
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_change_detection_timing_feedback
    counter = 3
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_change_detection, spatial_change_detection_timing_feedback, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = ctx_decision_making
    counter = 4.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = ctx_decision_making_feedback
    counter = 4.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(ctx_decision_making, ctx_decision_making_feedback, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    #plt.xlabel('Peak time diff. (ms)', fontsize=fs)
    plt.ylabel('Temporal Portion\nof Variance', fontsize=fs)
    plt.xticks([1,1+1.5,1+3,1+4.5], ['SR', 'COMP', 'CD', 'cue-DM'], fontsize=fs)

    #plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_ylim([0, 0.9])
    plt.text(0, 1.05, 'Control', color=color_list[0], fontsize=fs)
    plt.text(0, 0.95, 'Feedback', color=color_list[1], fontsize=fs)

    #plt.legend(loc=(0.1, 0.6), fontsize=fs, frameon=False)
    #fig.legend(loc='upper right', fontsize=fs, frameon=False)

    plt.show()
    return fig


fig=temporal_variance_by_feedback(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], ['w2_0.0_r2_0.0_signal2Strength_1.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/temporal_variance_by_feedback.pdf")


def temporal_variance_by_variable_delay(serial_idxes, serial_idxes_variable_delay):
    spatial_reproduction = nontiming_analysis_2.temporal_variance_spatial_reproduction(serial_idxes)
    spatial_reproduction_variable_delay = nontiming_analysis_2.temporal_variance_spatial_reproduction_variable_delay(serial_idxes_variable_delay)

    spatial_comparison = nontiming_analysis_2.temporal_variance_spatial_comparison(serial_idxes)
    spatial_comparison_variable_delay = nontiming_analysis_2.temporal_variance_spatial_comparison_variable_delay(serial_idxes_variable_delay)

    spatial_change_detection = nontiming_analysis_2.temporal_variance_spatial_change_detection(serial_idxes)
    spatial_change_detection_variable_delay = nontiming_analysis_2.temporal_variance_spatial_change_detection_variable_delay(serial_idxes_variable_delay)

    decision_making = nontiming_analysis_1.temporal_variance_decision_making(serial_idxes)
    decision_making_variable_delay = nontiming_analysis_1.temporal_variance_decision_making_variable_delay(serial_idxes)

    ctx_decision_making = nontiming_analysis_1.temporal_variance_ctx_decision_making(serial_idxes)
    ctx_decision_making_variable_delay = nontiming_analysis_1.temporal_variance_ctx_decision_making_variable_delay(serial_idxes)

    #print(stats.ttest_ind(spatial_reproduction, spatial_reproduction_variable_delay, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_comparison, spatial_comparison_variable_delay, equal_var=False)[1])
    #print(stats.ttest_ind(spatial_change_detection, spatial_change_detection_variable_delay, equal_var=False)[1])
    #print(stats.ttest_ind(decision_making, decision_making_variable_delay, equal_var=False)[1])
    #print(stats.ttest_ind(ctx_decision_making, ctx_decision_making_variable_delay, equal_var=False)[1])

    #############################################################################################
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.68, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    width = 0.5
    dots_random_width = 0.05
    single_result = np.array(spatial_reproduction)
    counter = 0
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_reproduction_variable_delay
    counter = 0
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))],
            width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_reproduction, spatial_reproduction_variable_delay, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_comparison
    counter = 1.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_comparison_variable_delay
    counter = 1.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result,  '.', color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_comparison, spatial_comparison_variable_delay, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = spatial_change_detection
    counter = 3
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = spatial_change_detection_variable_delay
    counter = 3
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(spatial_change_detection, spatial_change_detection_variable_delay, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = decision_making
    counter = 4.5
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = decision_making_variable_delay
    counter = 4.5
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(decision_making, decision_making_variable_delay, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    single_result = ctx_decision_making
    counter = 6
    x = np.random.normal(1+counter-width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[0], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter-width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[0], zorder=10)

    single_result = ctx_decision_making_variable_delay
    counter = 6
    x = np.random.normal(1+counter+width/2, dots_random_width, size=len(single_result))
    plt.plot(x, single_result, '.',  color=color_list[1], alpha=dot_alpha, zorder=0)
    plt.bar([1+counter+width/2], [np.mean(single_result)], yerr=[np.std(single_result)/np.sqrt(len(single_result))], width=width*0.92, fill=False, edgecolor=color_list[1], zorder=10)

    if stats.ttest_ind(ctx_decision_making, ctx_decision_making_variable_delay, equal_var=False)[1] < 0.05:
        ax.annotate('*', xy=(1+counter, 0.8), xytext=(1+counter, 0.82), xycoords='data',
                    fontsize=fs, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=1.5, lengthB=0.4', lw=1.0))

    #plt.xlabel('Peak time diff. (ms)', fontsize=fs)
    plt.ylabel('Temporal Portion\nof Variance', fontsize=fs)
    plt.xticks([1,1+1.5,1+3,1+4.5,1+6], ['SR', 'COMP', 'CD', 'DM', 'cue-DM'], fontsize=fs)

    #plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_ylim([0, 0.9])
    plt.text(0, 1.05, 'Control', color=color_list[0], fontsize=fs)
    plt.text(0, 0.95, 'Variable delay', color=color_list[1], fontsize=fs)

    #plt.legend(loc=(0.1, 0.6), fontsize=fs, frameon=False)
    #fig.legend(loc='upper right', fontsize=fs, frameon=False)

    plt.show()
    return fig


fig=temporal_variance_by_variable_delay(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], ['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/temporal_variance_by_variable_delay.pdf")

fig=nontiming_analysis_feedback.feedback_current(['w2_0.0_r2_0.0_signal2Strength_1.0/model_'+str(x)+'/finalResult' for x in range(16)])
fig.savefig("./figure/fig6/spatial_reproduction_feedback_current.pdf")