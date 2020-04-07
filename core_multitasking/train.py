"""Main training loop"""

from __future__ import division

import numpy as np
import torch
from torch.utils.data import DataLoader

from collections import defaultdict
import time
import sys
import os

from . import train_stepper
from . import network
from . import tools
from . import dataset

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']


def popvec_space(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def get_perf_space(output, target_time, target_dir, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, n_gaussianline=32.):

    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]+3:fix_end[i]-5, i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:, i, :] > action_threshold) == 0 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])
    rel_action_time = np.abs(action_time - target_time) / target_time

    direction = np.array([popvec_space(output[fix_end[i] + action_time[i], i, :]) for i in range(batch_size)]) / (2*np.pi) * n_gaussianline

    direction_err = np.abs(direction - target_dir)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_rel_action_time = np.mean(rel_action_time[np.argwhere(1 - fail_action)])
    mean_direction_err = np.mean(direction_err[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_rel_action_time, mean_direction_err


def get_perf_space_no_timing(output, target_dir, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20), n_gaussianline=32.):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]+3:fix_end[i]-5, i, :] > fix_strength) > 0 for i in range(batch_size)])

    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion

    #print(np.sum(action_at_fix)/batch_size, np.sum(no_action_at_motion)/batch_size)

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    direction = np.array([popvec_space(output[fix_end[i] + action_time[i], i, :]) for i in range(batch_size)]) / (2*np.pi) * n_gaussianline

    direction_err = np.abs(direction - target_dir)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_direction_err = np.mean(direction_err[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_direction_err


def get_perf_decision(output, target_time, target_choice, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+target_time[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    both_action_at_motion = np.array([np.sum(np.sum(output[fix_end[i]:fix_end[i]+target_time[i]+response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+target_time[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    rel_action_time = np.abs(action_time - target_time) / target_time

    action = np.concatenate([(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int) for i in range(batch_size)], axis=0)

    choice = action[:, 0] - action[:, 1]

    choice_correct = ((choice - target_choice) == 0)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_rel_action_time = np.mean(rel_action_time[np.argwhere(1 - fail_action)])
    mean_choice_correct = np.mean(choice_correct[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_rel_action_time, 1-mean_choice_correct


def get_perf_discrim(output, target_choice, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    both_action_at_motion = np.array([np.sum(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    action = np.concatenate([(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int) for i in range(batch_size)], axis=0)

    choice = action[:, 0] - action[:, 1]

    choice_correct = ((choice - target_choice) == 0)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_choice_correct = np.mean(choice_correct[np.argwhere(1 - fail_action)])

    return success_action_prob, 1-mean_choice_correct


def get_perf_space_comp(output, target_choice, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    both_action_at_motion = np.array([np.sum(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    action = np.concatenate([(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int) for i in range(batch_size)], axis=0)

    choice = action[:, 0] - action[:, 1]

    choice_correct = ((choice - target_choice) == 0)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_choice_correct = np.mean(choice_correct[np.argwhere(1 - fail_action)])

    return success_action_prob, 1-mean_choice_correct


def get_perf_direction_no_timing(output, target_dir, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    direction = np.array([popvec(output[fix_end[i] + action_time[i], i, :]) for i in range(batch_size)])

    direction_err = direction - target_dir
    direction_err = np.minimum(np.abs(direction_err), 2*np.pi-np.abs(direction_err))

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_direction_err = np.mean(direction_err[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_direction_err


def get_perf_decision_no_timing(output, target_choice, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5, response_duration=int(300/20)):
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i, :] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    both_action_at_motion = np.array([np.sum(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=0) > 0) > 1 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion + both_action_at_motion

    action_time = np.array([np.argmax(np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold, axis=1) > 0) for i in range(batch_size)])

    action = np.concatenate([(output[[fix_end[i] + action_time[i]], i, :] > action_threshold).astype(np.int) for i in range(batch_size)], axis=0)

    choice = action[:, 0] - action[:, 1]

    choice_correct = ((choice - target_choice) == 0)

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_choice_correct = np.mean(choice_correct[np.argwhere(1 - fail_action)])

    return success_action_prob, 1-mean_choice_correct


class Trainer(object):
    '''
    rule_name_list: [rule1, rule2]
    '''
    def __init__(self, rule_name_list=None, model=None, hp=None, model_dir=None, is_cuda=True, **kwargs):
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir

        self.rule_name_list = rule_name_list
        self.is_cuda = is_cuda

        if is_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # load or create hyper-parameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # hyper-parameters for time scale
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        self.hp = hp

        fh_fname = os.path.join(model_dir, 'hp.json')
        if not os.path.isfile(fh_fname):
            tools.save_hp(hp, model_dir)

        kwargs['rule_name'] = rule_name_list[0]
        # load or create model
        if model is None:
            self.model = network.RNN(hp, is_cuda, **kwargs)
            self.model.load(model_dir)
        else:
            self.model = model

        # load or create log
        self.log = tools.load_log(model_dir)
        if self.log is None:
            self.log = defaultdict(list)
            self.log['model_dir'] = model_dir

        # trainner stepper
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda)

        # collate_fn of dataloader
        def collate_fn(batch):
            return batch[0]

        #print(rule_name)
        #print(type(rule_name))
        del kwargs['rule_name']
        # data loader
        dataset_train_rule0 = dataset.TaskDataset(rule_name_list[0], hp, mode='train', is_cuda=is_cuda, **kwargs)
        dataset_test_rule0 = dataset.TaskDataset(rule_name_list[0], hp, mode='test', is_cuda=is_cuda, **kwargs)
        dataset_train_rule1 = dataset.TaskDataset(rule_name_list[1], hp, mode='train', is_cuda=is_cuda, **kwargs)
        dataset_test_rule1 = dataset.TaskDataset(rule_name_list[1], hp, mode='test', is_cuda=is_cuda, **kwargs)

        self.dataloader_train_rule0 = DataLoader(dataset_train_rule0, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        self.dataloader_test_rule0 = DataLoader(dataset_test_rule0, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        self.dataloader_train_rule1 = DataLoader(dataset_train_rule1, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        self.dataloader_test_rule1 = DataLoader(dataset_test_rule1, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

        self.min_cost = np.inf

        self.model_save_idx = 0
        self.info = [dict(), dict()]

    def do_eval(self, rule_index):
        '''Do evaluation, and then save the model
        '''

        print('Trial {:7d}'.format(self.log['trials'][-1]) +
              '  | Time {:0.2f} s'.format(self.log['times'][-1]) +
              '  | rule id {:7d}'.format(rule_index))

        if rule_index == 0:
            dataloader_test = self.dataloader_test_rule0
            print(id(dataloader_test) == id(self.dataloader_test_rule0))
            rule_name = self.rule_name_list[0]
        else:
            dataloader_test = self.dataloader_test_rule1
            print(id(dataloader_test) == id(self.dataloader_test_rule1))
            rule_name = self.rule_name_list[1]

        for i_batch, sample_batched in enumerate(dataloader_test):
            '''training'''

            clsq_tmp = list()
            creg_tmp = list()

            if self.is_cuda:
                sample_batched['inputs'] = sample_batched['inputs'].cuda()
                sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

            sample_batched['rule_name'] = rule_name

            with torch.no_grad():
                self.train_stepper.cost_fcn(**sample_batched)

            clsq_tmp.append(self.train_stepper.cost_lsq.detach().cpu().numpy())
            creg_tmp.append(self.train_stepper.cost_reg.detach().cpu().numpy())

            self.log['cost_rule'+str(rule_index)].append(np.mean(clsq_tmp, dtype=np.float64))
            self.log['creg_rule'+str(rule_index)].append(np.mean(creg_tmp, dtype=np.float64))

            # log['perf_' + rule_test].append(np.mean(perf_tmp, dtype=np.float64))
            print('| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
                  '| c_reg {:0.6f}'.format(np.mean(creg_tmp)))

            sys.stdout.flush()

            if clsq_tmp[-1] < self.min_cost:
                self.min_cost = clsq_tmp[-1]
                # Saving the model and log
                print('save model!')
                self.model.save(self.model_dir)

            tools.save_log(self.log)

            # save model routinely
            routine_save_path = os.path.join(self.model_dir, str(self.model_save_idx))

            tools.mkdir_p(routine_save_path)
            self.model.save(routine_save_path)

            # save performance
            if rule_name == 'timed_decision_making':
                success_action_prob, mean_rel_action_time, mean_choice_error = get_perf_decision(self.train_stepper.outputs.detach().cpu().numpy(), sample_batched['prod_interval'], np.sign(sample_batched['strength1']-sample_batched['strength2']), sample_batched['epochs']['interval'][0], sample_batched['epochs']['go_cue'][1])
                success_action_prob = success_action_prob.tolist()
                mean_rel_action_time = mean_rel_action_time.tolist()
                mean_choice_error = mean_choice_error.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_rel_action_time'] = mean_rel_action_time
                self.info[rule_index]['mean_choice_error'] = mean_choice_error
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_rel_action_time {:0.6f}'.format(mean_rel_action_time) +
                      '| mean_choice_error {:0.6f}'.format(mean_choice_error))

            elif rule_name == 'decision_making':
                success_action_prob, mean_choice_error = get_perf_decision_no_timing(self.train_stepper.outputs.detach().cpu().numpy(), np.sign(sample_batched['strength1']-sample_batched['strength2']), sample_batched['epochs']['interval'][0], sample_batched['epochs']['interval'][1])
                success_action_prob = success_action_prob.tolist()
                mean_choice_error = mean_choice_error.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_choice_error'] = mean_choice_error
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_choice_error {:0.6f}'.format(mean_choice_error))

            elif rule_name == 'timed_spatial_reproduction':
                success_action_prob, mean_rel_action_time, mean_direction_err = get_perf_space(self.train_stepper.outputs.detach().cpu().numpy(), sample_batched['prod_interval'], sample_batched['gaussian_center'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1], n_gaussianline=32.)
                success_action_prob = success_action_prob.tolist()
                mean_rel_action_time = mean_rel_action_time.tolist()
                mean_direction_err = mean_direction_err.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_rel_action_time'] = mean_rel_action_time
                self.info[rule_index]['mean_direction_err'] = mean_direction_err
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_rel_action_time {:0.6f}'.format(mean_rel_action_time) +
                      '| mean_direction_err {:0.6f}'.format(mean_direction_err))

            elif rule_name == 'timed_spatial_reproduction_broad_tuning':
                success_action_prob, mean_rel_action_time, mean_direction_err = get_perf_space(self.train_stepper.outputs.detach().cpu().numpy(), sample_batched['prod_interval'], sample_batched['gaussian_center'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1], n_gaussianline=44.)
                success_action_prob = success_action_prob.tolist()
                mean_rel_action_time = mean_rel_action_time.tolist()
                mean_direction_err = mean_direction_err.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_rel_action_time'] = mean_rel_action_time
                self.info[rule_index]['mean_direction_err'] = mean_direction_err
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_rel_action_time {:0.6f}'.format(mean_rel_action_time) +
                      '| mean_direction_err {:0.6f}'.format(mean_direction_err))

            elif rule_name == 'spatial_reproduction':
                success_action_prob,  mean_direction_err = get_perf_space_no_timing(self.train_stepper.outputs.detach().cpu().numpy(), sample_batched['gaussian_center'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1], n_gaussianline=32.)
                success_action_prob = success_action_prob.tolist()
                mean_direction_err = mean_direction_err.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_direction_err'] = mean_direction_err
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_frequency_err {:0.6f}'.format(mean_direction_err))

            elif rule_name == 'spatial_reproduction_broad_tuning':
                success_action_prob,  mean_direction_err = get_perf_space_no_timing(self.train_stepper.outputs.detach().cpu().numpy(), sample_batched['gaussian_center'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1], n_gaussianline=44.)
                success_action_prob = success_action_prob.tolist()
                mean_direction_err = mean_direction_err.tolist()

                self.info[rule_index] = dict()
                self.info[rule_index]['cost'] = clsq_tmp[-1].tolist()
                self.info[rule_index]['creg'] = creg_tmp[-1].tolist()
                self.info[rule_index]['success_action_prob'] = success_action_prob
                self.info[rule_index]['mean_direction_err'] = mean_direction_err
                self.info[rule_index]['model_dir'] = routine_save_path
                log_name = 'log_rule'+str(rule_index)+'.json'
                tools.save_log(self.info[rule_index], log_name=log_name)
                tools.save_hp(self.hp, routine_save_path)

                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_frequency_err {:0.6f}'.format(mean_direction_err))

            if i_batch == 0:
                if rule_name == 'timed_decision_making':
                    return clsq_tmp[-1], success_action_prob, mean_rel_action_time, mean_choice_error
                elif rule_name == 'decision_making':
                    return clsq_tmp[-1], success_action_prob, mean_choice_error
                elif rule_name == 'timed_spatial_reproduction':
                    return clsq_tmp[-1], success_action_prob, mean_rel_action_time, mean_direction_err
                elif rule_name == 'timed_spatial_reproduction_broad_tuning':
                    return clsq_tmp[-1], success_action_prob, mean_rel_action_time, mean_direction_err
                elif rule_name == 'spatial_reproduction':
                    return clsq_tmp[-1], success_action_prob, mean_direction_err
                elif rule_name == 'spatial_reproduction_broad_tuning':
                    return clsq_tmp[-1], success_action_prob, mean_direction_err

    def save_final_result(self):
        save_path = os.path.join(self.model_dir, 'finalResult')
        tools.mkdir_p(save_path)
        self.model.save(save_path)
        self.info[0]['model_dir'] = save_path
        log_name = 'log_rule'+str(0)+'.json'
        tools.save_log(self.info[0], log_name=log_name)
        self.info[1]['model_dir'] = save_path
        log_name = 'log_rule'+str(1)+'.json'
        tools.save_log(self.info[1], log_name=log_name)
        tools.save_hp(self.hp, save_path)

    def train(self, max_samples=1e7, display_step=500, max_model_save_idx=150):
        """Train the network.

        Args:
            max_sample: int, maximum number of training samples
            display_step: int, display steps
        Returns:
            model is stored at model_dir/model.ckpt
            training configuration is stored at model_dir/hp.json
        """

        # Display hp
        for key, val in self.hp.items():
            print('{:20s} = '.format(key) + str(val))

        # Record time
        t_start = time.time()
        for (step_rule0, sample_batched_rule0), (step_rule1, sample_batched_rule1) in zip(enumerate(self.dataloader_train_rule0), enumerate(self.dataloader_train_rule1)):
            try:
                # train rule0
                if self.is_cuda:

                    sample_batched_rule0['inputs'] = sample_batched_rule0['inputs'].cuda()
                    sample_batched_rule0['target_outputs'] = sample_batched_rule0['target_outputs'].cuda()
                    sample_batched_rule0['cost_mask'] = sample_batched_rule0['cost_mask'].cuda()
                    sample_batched_rule0['seq_mask'] = sample_batched_rule0['seq_mask'].cuda()
                    sample_batched_rule0['initial_state'] = sample_batched_rule0['initial_state'].cuda()

                sample_batched_rule0['rule_name'] = self.rule_name_list[0]

                if self.model_save_idx < 5:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(1e-3, device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(1e-3, device=self.device)
                else:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(self.hp['l2_firing_rate'], device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(self.hp['l2_firing_rate'], device=self.device)

                self.train_stepper.stepper(**sample_batched_rule0)

                # train rule1
                if self.is_cuda:

                    sample_batched_rule1['inputs'] = sample_batched_rule1['inputs'].cuda()
                    sample_batched_rule1['target_outputs'] = sample_batched_rule1['target_outputs'].cuda()
                    sample_batched_rule1['cost_mask'] = sample_batched_rule1['cost_mask'].cuda()
                    sample_batched_rule1['seq_mask'] = sample_batched_rule1['seq_mask'].cuda()
                    sample_batched_rule1['initial_state'] = sample_batched_rule1['initial_state'].cuda()

                sample_batched_rule1['rule_name'] = self.rule_name_list[1]

                if self.model_save_idx < 5:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(1e-3, device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(1e-3, device=self.device)
                else:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(self.hp['l2_firing_rate'], device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(self.hp['l2_firing_rate'], device=self.device)

                self.train_stepper.stepper(**sample_batched_rule1)
                ######################################################################
                if step_rule0 % display_step == 0:

                    self.log['trials'].append(step_rule0 * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)

                    if self.rule_name_list[0] == 'timed_decision_making':
                        cost_rule0, success_action_prob_rule0, mean_rel_action_time_rule0, mean_choice_err_rule0 = self.do_eval(0)
                        cost_rule1, success_action_prob_rule1, mean_choice_err_rule1 = self.do_eval(1)
                        self.model_save_idx = self.model_save_idx + 1
                        if not np.isfinite(cost_rule0):
                            return 'error'
                        if not np.isfinite(cost_rule1):
                            return 'error'

                        if success_action_prob_rule0 > 0.9 and mean_rel_action_time_rule0 < 0.025 and mean_choice_err_rule0 < 0.02 \
                                and success_action_prob_rule1 > 0.9 and mean_choice_err_rule1 < 0.02:
                            self.save_final_result()
                            break

                        if self.model_save_idx > max_model_save_idx:
                            break

                    elif self.rule_name_list[0] == 'timed_spatial_reproduction':
                        cost0, success_action_prob0, mean_rel_action_time0, mean_frequency_err0 = self.do_eval(0)
                        cost1, success_action_prob1, mean_frequency_err1 = self.do_eval(1)
                        self.model_save_idx = self.model_save_idx + 1

                        if not np.isfinite(cost0):
                            return 'error'
                        if not np.isfinite(cost1):
                            return 'error'

                        if success_action_prob0 > 0.95 and mean_rel_action_time0 < 0.03 and mean_frequency_err0 < 0.8 \
                                and success_action_prob1 > 0.95 and mean_frequency_err1 < 0.8:
                            self.save_final_result()
                            break

                        if self.model_save_idx > max_model_save_idx:
                            break

                    elif self.rule_name_list[0] == 'timed_spatial_reproduction_broad_tuning':
                        cost0, success_action_prob0, mean_rel_action_time0, mean_frequency_err0 = self.do_eval(0)
                        cost1, success_action_prob1, mean_frequency_err1 = self.do_eval(1)
                        self.model_save_idx = self.model_save_idx + 1

                        if not np.isfinite(cost0):
                            return 'error'
                        if not np.isfinite(cost1):
                            return 'error'

                        if success_action_prob0 > 0.95 and mean_rel_action_time0 < 0.03 and mean_frequency_err0 < 0.8 \
                                and success_action_prob1 > 0.95 and mean_frequency_err1 < 0.8:
                            self.save_final_result()
                            break

                        if self.model_save_idx > max_model_save_idx:
                            break

                if step_rule0 * self.hp['batch_size_train'] > max_samples:
                    self.log['trials'].append(step_rule0 * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    self.do_eval(0)
                    self.do_eval(1)

                    break

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")

        return 'OK'
