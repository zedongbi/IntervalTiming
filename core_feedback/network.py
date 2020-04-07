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
        input_feedback_size = hp['n_input_feedback']

        hidden_size = hp['n_rnn']
        output_size = hp['n_output']
        alpha = hp['alpha']
        sigma_rec = hp['sigma_rec']
        act_fcn = hp['activation']

        self.hp = hp

        self.input_size = input_size
        self.input_feedback_size = input_feedback_size
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

        self.act_fcn_sensory = lambda x: nn.functional.relu(x)
        self.act_fcn_feedback = lambda x,y: torch.min(x, y)

        self.sensory_threshold = torch.tensor(1.5, device=self.device)


        if kwargs['rule_name'] == 'spatial_reproduction':
            if input_size is not 33 or input_feedback_size is not 1:
                raise Exception('input_size=33,  input_feedback_size=1, for spatial_reproduction')
            # feedforward connection
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(4.), 1./math.sqrt(4.))
            weight_ih[0:input_feedback_size, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)
            # feedback connection
            self.weight_hi = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(1.3/hidden_size, 1.7/hidden_size))
            # feedback mask, the first input_feedback_size units receives feedback
            self.feedback_mask = torch.zeros((1, input_size), device=self.device)
            self.feedback_mask[:, :input_feedback_size] = 1.
            # feedforward threshold, set the threshold of the units without feedback to zero
            self.sensory_threshold_tensor = torch.ones((1, input_size), device=self.device) * self.sensory_threshold
            self.sensory_threshold_tensor[:, input_feedback_size:] = 0

        elif kwargs['rule_name'] == 'spatial_comparison':
            if input_size is not 32+32 or input_feedback_size is not 32:
                raise Exception('input_size=64,  input_feedback_size=32, for spatial_comparison')
            # feedforward connection
            self.weight_ih = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(4.), 1./math.sqrt(4.)))
            # feedback connection
            self.weight_hi = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(1.3/hidden_size, 1.7/hidden_size))
            # feedback mask, the first input_feedback_size units receives feedback
            self.feedback_mask = torch.zeros((1, input_size), device=self.device)
            self.feedback_mask[:, :input_feedback_size] = 1.
            # feedforward threshold, set the threshold of the units without feedback to zero
            self.sensory_threshold_tensor = torch.ones((1, input_size), device=self.device) * self.sensory_threshold
            self.sensory_threshold_tensor[:, input_feedback_size:] = 0

        elif kwargs['rule_name'] == 'spatial_change_detection':
            if input_size is not 32+32 or input_feedback_size is not 32:
                raise Exception('input_size=64,  input_feedback_size=32, for spatial_change_detection')
            # feedforward connection
            self.weight_ih = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(4.), 1./math.sqrt(4.)))
            # feedback connection
            self.weight_hi = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(1.3/hidden_size, 1.7/hidden_size))
            # feedback mask, the first input_feedback_size units receives feedback
            self.feedback_mask = torch.zeros((1, input_size), device=self.device)
            self.feedback_mask[:, :input_feedback_size] = 1.
            # feedforward threshold, set the threshold of the units without feedback to zero
            self.sensory_threshold_tensor = torch.ones((1, input_size), device=self.device) * self.sensory_threshold
            self.sensory_threshold_tensor[:, input_feedback_size:] = 0

        elif kwargs['rule_name'] == 'ctx_decision_making':
            if input_size is not 4 or input_feedback_size is not 2:
                raise Exception('input_size=4,  input_feedback_size=2, for ctx_decision_making')
            # feedforward connection
            weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
            weight_ih[0:input_feedback_size, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
            self.weight_ih = nn.Parameter(weight_ih)
            # feedback connection
            self.weight_hi = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(1.3/hidden_size, 1.7/hidden_size))
            # feedback mask, the first input_feedback_size units receives feedback
            self.feedback_mask = torch.zeros((1, input_size), device=self.device)
            self.feedback_mask[:, :input_feedback_size] = 1.
            # feedforward threshold, set the threshold of the units without feedback to zero
            self.sensory_threshold_tensor = torch.ones((1, input_size), device=self.device) * self.sensory_threshold
            self.sensory_threshold_tensor[:, input_feedback_size:] = 0

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

        """Most basic RNN: output = new_state = W_input * input + W_rec * act(state) + B + noise """

        # shape: (batch_size, hidden_size)
        state = initial_state
        state_collector = [state]
        feedback_current_collector = []

        for input_per_step in inputs:
            feed_back_current = torch.matmul(self.act_fcn(state), self.weight_hi) * self.feedback_mask
            feedback_current_collector.append(feed_back_current[:, :self.input_feedback_size])

            feed_back = self.act_fcn_feedback(feed_back_current, self.sensory_threshold)

            state_new = torch.matmul(self.act_fcn(state), self.weight_hh) + self.bias_h + \
                        torch.matmul(self.act_fcn_sensory(input_per_step + feed_back - self.sensory_threshold_tensor), self.weight_ih) + torch.randn_like(state, device=self.device) * self.sigma_rec

            state = (self._1 - self.alpha) * state + self.alpha * state_new
            state_collector.append(state)

        return state_collector, feedback_current_collector

    def feedback_weight_clipper(self):
        self.weight_hi.data.clamp_(0.)

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
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)


