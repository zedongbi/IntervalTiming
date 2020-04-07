"""Definition of the network model and various RNN cells"""
from __future__ import division

import torch
from torch import nn

import os
import math


# Create Network
class RNN(nn.Module):
    def __init__(self, hp, is_cuda=True, **kwargs):
        super(RNN, self).__init__()

        input_size = hp['n_input']
        hidden_size = hp['n_rnn']
        output_size = hp['n_output']
        alpha = hp['alpha']
        sigma_rec = hp['sigma_rec']
        act_fcn = hp['activation']

        self.hp = hp

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if is_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self._clamp_thres = torch.tensor(100000000., device=self.device)
        self._high_value_coeff = torch.tensor(0.5, device=self.device)

        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)


        if kwargs['rule_name'] == 'timed_decision_making':
            if input_size is not 5:
                raise Exception('input_size should be 5 for timed_decision_making')
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            weight_ih[2, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            weight_ih[3:5, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)

        elif kwargs['rule_name'] == 'decision_making':
            if input_size is not 5:
                raise Exception('input_size should be 5 for timed_decision_making')
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            weight_ih[2, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            weight_ih[3:5, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)

        elif kwargs['rule_name'] == 'timed_spatial_reproduction':
            if input_size is not 34+2:
                raise Exception('input_size should be 36 for timed_spatial_reproduction')
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(4.), 1./math.sqrt(4.))
            weight_ih[0:2, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)

        elif kwargs['rule_name'] == 'spatial_reproduction':
            if input_size is not 34+2:
                raise Exception('input_size should be 36 for spatial_reproduction')
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(4.), 1./math.sqrt(4.))
            weight_ih[0:1, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)


        hh_mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
        non_diag = torch.empty(hidden_size, hidden_size).normal_(0, hp['initial_std']/math.sqrt(hidden_size))
        weight_hh = torch.eye(hidden_size)*0.999 + hh_mask * non_diag

        self.weight_hh = nn.Parameter(weight_hh)
        self.bias_h = nn.Parameter(torch.zeros(1, hidden_size))

        self.weight_out = nn.Parameter(torch.empty(hidden_size, output_size).normal_(0., 0.4/math.sqrt(hidden_size)))
        self.bias_out = nn.Parameter(torch.zeros(output_size,))

        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2./alpha) * sigma_rec, device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

    def forward(self, inputs, initial_state):

        """Most basic RNN: output = new_state = W_input * input + W_rec * act(state) + B + noise"""

        # shape: (batch_size, hidden_size)
        state = initial_state
        state_collector = [state]

        for input_per_step in inputs:
            state_new = torch.matmul(self.act_fcn(state), self.weight_hh) + self.bias_h + \
                        torch.matmul(input_per_step, self.weight_ih) + torch.randn_like(state, device=self.device) * self.sigma_rec

            state = (self._1 - self.alpha) * state + self.alpha * state_new
            state_collector.append(state)

        return state_collector

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element = self.weight_hh.diag().data.clamp_(0., 1.)
        self.weight_hh.data[range(self.hidden_size), range(self.hidden_size)] = diag_element

    def save(self, model_dir):
        save_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.state_dict(), save_path)

    def load(self, model_dir):
        if model_dir is not None:
            save_path = os.path.join(model_dir, 'model.pth')
            if os.path.isfile(save_path):
                #self.load_state_dict(torch.load(save_path), strict=False)
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)

