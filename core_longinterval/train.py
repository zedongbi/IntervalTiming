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


def get_perf_prod(output, target_time, fix_start, fix_end, fix_strength=0.2, action_threshold=0.5):

    batch_size = output.shape[1]

    action_at_fix = np.array([np.sum(output[fix_start[i]:fix_end[i], i] > fix_strength) > 0 for i in range(batch_size)])
    no_action_at_motion = np.array([np.sum(output[fix_end[i]:, i] > action_threshold) == 0 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion
    action_time = np.array([np.argmax(output[fix_end[i]:, i] > action_threshold) for i in range(batch_size)])
    rel_action_time = np.abs(action_time - target_time) / target_time

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_rel_action_time = np.mean(rel_action_time[np.argwhere(1 - fail_action)])

    return success_action_prob, mean_rel_action_time


class Trainer(object):
    def __init__(self, rule_name=None, model=None, hp=None, model_dir=None, is_cuda=True, **kwargs):
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir

        self.rule_name = rule_name
        self.is_cuda = is_cuda

        if is_cuda:
            self.device = torch.device("cuda")
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

        kwargs['rule_name'] = rule_name
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
        dataset_train = dataset.TaskDataset(rule_name, hp, mode='train', is_cuda=is_cuda, **kwargs)
        dataset_test = dataset.TaskDataset(rule_name, hp, mode='test', is_cuda=is_cuda, **kwargs)

        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

        self.min_cost = np.inf

        self.model_save_idx = 0

    def do_eval(self):
        '''Do evaluation, and then save the model
        '''

        print('Trial {:7d}'.format(self.log['trials'][-1]) +
              '  | Time {:0.2f} s'.format(self.log['times'][-1]))

        for i_batch, sample_batched in enumerate(self.dataloader_test):
            '''training'''

            clsq_tmp = list()
            creg_tmp = list()

            if self.is_cuda:
                sample_batched['inputs'] = sample_batched['inputs'].cuda()
                sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

            sample_batched['rule_name'] = self.rule_name

            with torch.no_grad():
                self.train_stepper.cost_fcn(**sample_batched)

            clsq_tmp.append(self.train_stepper.cost_lsq.detach().cpu().numpy())
            creg_tmp.append(self.train_stepper.cost_reg.detach().cpu().numpy())

            self.log['cost_'].append(np.mean(clsq_tmp, dtype=np.float64))
            self.log['creg_'].append(np.mean(creg_tmp, dtype=np.float64))

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
            self.model_save_idx = self.model_save_idx + 1
            tools.mkdir_p(routine_save_path)
            self.model.save(routine_save_path)

            # save performance
            if self.rule_name == 'interval_production_long_interval':
                success_action_prob, mean_rel_action_time = get_perf_prod(self.train_stepper.outputs[:, :, 0].detach().cpu().numpy(), sample_batched['prod_interval'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1])
                success_action_prob = success_action_prob.tolist()
                mean_rel_action_time = mean_rel_action_time.tolist()

                self.info = dict()
                self.info['cost'] = clsq_tmp[-1].tolist()
                self.info['creg'] = creg_tmp[-1].tolist()
                self.info['success_action_prob'] = success_action_prob
                self.info['mean_rel_action_time'] = mean_rel_action_time
                self.info['model_dir'] = routine_save_path
                tools.save_log(self.info)
                tools.save_hp(self.hp, routine_save_path)
                print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                      '| mean_rel_action_time {:0.6f}'.format(mean_rel_action_time))

            if i_batch == 0:
                if self.rule_name == 'interval_production_long_interval':
                    return clsq_tmp[-1], success_action_prob, mean_rel_action_time

    def save_final_result(self):
        save_path = os.path.join(self.model_dir, 'finalResult')
        tools.mkdir_p(save_path)
        self.model.save(save_path)
        self.info['model_dir'] = save_path
        tools.save_log(self.info)
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
        for step, sample_batched in enumerate(self.dataloader_train):
            try:
                if self.is_cuda:

                    sample_batched['inputs'] = sample_batched['inputs'].cuda()
                    sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda()
                    sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda()
                    sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda()
                    sample_batched['initial_state'] = sample_batched['initial_state'].cuda()

                sample_batched['rule_name'] = self.rule_name

                if self.model_save_idx < 5:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(1e-3, device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(1e-3, device=self.device)
                else:
                    self.train_stepper.l2_firing_rate_cpu = torch.tensor(self.hp['l2_firing_rate'], device=torch.device("cpu"))
                    self.train_stepper.l2_firing_rate = torch.tensor(self.hp['l2_firing_rate'], device=self.device)

                self.train_stepper.stepper(**sample_batched)

                if step % display_step == 0:

                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    if self.rule_name == 'interval_production_long_interval':
                        cost, success_action_prob, mean_rel_action_time = self.do_eval()
                        if not np.isfinite(cost):
                            return 'error'
                        if success_action_prob > 0.95 and mean_rel_action_time < 0.025:
                            self.save_final_result()
                            break
                        if self.model_save_idx > max_model_save_idx:
                            break

                if step * self.hp['batch_size_train'] > max_samples:
                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    self.do_eval()
                    break

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")

        return 'OK'
