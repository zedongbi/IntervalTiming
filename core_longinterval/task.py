"""Collections of tasks."""

from __future__ import division
import numpy as np
import math


# generating values of x, y, c_mask specific for tasks
# config contains hyper-parameters used for generating tasks
class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, xtdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            xtdim: int, number of total time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.xtdim = xtdim

        # time major
        self.x = np.zeros((xtdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        self.cost_mask = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        # strength of input noise
        self._sigma_x = config['sigma_x'] * math.sqrt(2./self.config['alpha'])

        if config['rule_name'] == 'timed_spatial_reproduction_broad_tuning' \
                or config['rule_name'] == 'spatial_reproduction_broad_tuning' \
                or config['rule_name'] == 'spatial_comparison_broad_tuning' \
                or config['rule_name'] == 'spatial_change_detection_broad_tuning':
            self.n_guassianline = 32 + 12
            self.sd_gaussianline = 4.
        else:
            self.n_guassianline = 32
            self.sd_gaussianline = 2.

        self.pref_line_gaussian = np.arange(0, self.n_guassianline)

    def line_gaussian_activity(self, x_loc):
        """Input activity given location."""
        dist = np.abs(x_loc - self.pref_line_gaussian)  # non_periodic boundary
        dist /= self.sd_gaussianline  # standard deviation
        return np.exp(-dist ** 2 / 2)

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths, gaussian_center=None):
        """Add an input or stimulus output to the indicated channel.

        Args:
            loc_type: str type of information to be added
            loc_idx: index of channel
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
            gaussian_center: float. location of gaussian bump, only used if loc_type=='line_gaussian_input' or 'line_gaussian_output'
        """

        if loc_type == 'input':
            for i in range(self.batch_size):
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'out':

            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'cost_mask':

            for i in range(self.batch_size):
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'line_gaussian_input':

            loc_start_idx = loc_idx
            loc_end_idx = loc_idx + self.n_guassianline

            for i in range(self.batch_size):
                self.x[ons[i]: offs[i], i, loc_start_idx:loc_end_idx] = self.line_gaussian_activity(gaussian_center[i]) * strengths[i]

        elif loc_type == 'line_gaussian_output':

            loc_start_idx = loc_idx
            loc_end_idx = loc_idx + self.n_guassianline

            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_start_idx:loc_end_idx] = self.line_gaussian_activity(gaussian_center[i]) * strengths[i]

        else:
            raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x


# basic timing tasks
def _interval_production_long_interval(config, mode, **kwargs):
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt) #int(200/dt) #int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(400, 2800, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        prod_interval = (rng.uniform(600, 2400, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)

        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)  # int(rng.choice([100, 200])/dt)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)  # int(rng.choice([100, 200])/dt)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + prod_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # the onset time of the go cue
    control_on = stim2_off + dly_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration
    # response start time
    response_on = control_off + prod_interval
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # pulse input
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))

    # go cue
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))

    # output
    trial.add('out', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'delay': (stim2_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.dly_interval = dly_interval
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def interval_production_long_interval(config, mode, **kwargs):
    return _interval_production_long_interval(config, mode, **kwargs)


# map string to function
rule_mapping = {
                'interval_production_long_interval': interval_production_long_interval
                }


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    # print(rule)
    config = hp
    kwargs['noise_on'] = noise_on
    trial = rule_mapping[rule](config, mode, **kwargs)

    if noise_on:
        trial.add_x_noise()

    return trial

