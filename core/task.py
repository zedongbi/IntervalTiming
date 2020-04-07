"""Collections of tasks."""

from __future__ import division
import numpy as np
import math
import sys


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))


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

def _interval_production(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)

    :param mode: the mode of generating. Options: 'random', 'random_validate', 'test'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param prod_interval, dly_interval: used only for 'test' mod

    :return: Trial object
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt) #int(200/dt) #int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(600, 1200, batch_size)/dt).astype(int)
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


def interval_production(config, mode, **kwargs):
    return _interval_production(config, mode, **kwargs)


def _interval_comparison(config, mode, **kwargs):
    '''
    Two stimuli are successively presented, the network should indicate which one has a longer duration
    '''
    dt = config['dt']
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # print('batch_size: ', batch_size)

        prod_interval1 = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
        prod_interval2 = (rng.uniform(400, 1400, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # print('batch_size: ', batch_size)

        prod_interval1 = (rng.uniform(600, 1200, batch_size)/dt).astype(int)
        prod_interval2 = (rng.uniform(600, 1200, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval1 = kwargs['prod_interval1']
        if not hasattr(prod_interval1, '__iter__'):
            prod_interval1 = np.array([prod_interval1] * batch_size)
        prod_interval1 = (prod_interval1 / dt).astype(int)

        prod_interval2 = kwargs['prod_interval2']
        if not hasattr(prod_interval2, '__iter__'):
            prod_interval2 = np.array([prod_interval2] * batch_size)
        prod_interval2 = (prod_interval2 / dt).astype(int)

        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(100, 500, batch_size)/dt).astype(int)  # int(rng.choice([100, 200])/dt)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)  # int(rng.choice([100, 200])/dt)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval1
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + prod_interval2
    # total time steps
    xtdim = stim2_off + response_duration   # int(200 / dt)

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    trial = Trial(config, xtdim.max(), batch_size)
    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.))
    trial.add('input', 1, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))

    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=xtdim, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'delay': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, xtdim)}

    trial.prod_interval1 = prod_interval1
    trial.prod_interval2 = prod_interval2

    trial.dly_interval = dly_interval

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def interval_comparison(config, mode, **kwargs):
    return _interval_comparison(config, mode, **kwargs)


# combined timing tasks
def _timed_spatial_reproduction(config, mode, **kwargs):
    '''
    Two pulses with duration T in-between are successively input to the network, with the first pulse contains spatial information.
    After a third pulse, the network should response toward the spatial location at time T after the third pulse.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(400, 1400, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(600, 1200, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

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

        gaussian_center = kwargs['gaussian_center']
        if not hasattr(gaussian_center, '__iter__'):
            gaussian_center = np.array([gaussian_center] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

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
    trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))
    # go cue
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 2, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)


    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
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
    trial.gaussian_center = gaussian_center
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def timed_spatial_reproduction(config, mode, **kwargs):
    return _timed_spatial_reproduction(config, mode, **kwargs)


def _timed_spatial_reproduction_broad_tuning(config, mode, **kwargs):
    '''
    timed spatial reproduction with broadened Gaussian tuning curve
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(400, 1400, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(12., 32+12-12.), batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(600, 1200, batch_size)/dt).astype(int)

        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(12., 32+12-12.), batch_size)

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

        gaussian_center = kwargs['gaussian_center']
        if not hasattr(gaussian_center, '__iter__'):
            gaussian_center = np.array([gaussian_center] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

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
    trial.add('input', 0, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.))
    # go cue
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 2, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
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
    trial.gaussian_center = gaussian_center

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial



def timed_spatial_reproduction_broad_tuning(config, mode, **kwargs):
    return _timed_spatial_reproduction_broad_tuning(config, mode, **kwargs)


def _timed_decision_making(config, mode, **kwargs):
    '''
    Two stimuli are presented concurrently with duration T. At time T after a pulse, the network should indicate which stimulus is stronger.
    '''
    dt = config['dt']
    pulse_duration = int(60/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(400, 1400, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(600, 1200, batch_size)/dt).astype(int)
        dly_interval = (rng.uniform(600, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

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

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval

    cue_on = stim1_off + dly_interval
    cue_off = cue_on + pulse_duration

    response_on = cue_off + prod_interval
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # cue
    trial.add('input', 2, ons=cue_on, offs=cue_off, strengths=trial.expand(1.))
    # output
    output_target1 = 1. * (strength1 > strength2)
    output_target2 = 1. * (strength1 <= strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'interval': (stim1_on, stim1_off),
                    'delay': (stim1_off, cue_on),
                    'go_cue': (cue_on, cue_off),
                    'go': (cue_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.dly_interval = dly_interval
    trial.strength1 = strength1
    trial.strength2 = strength2

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def timed_decision_making(config, mode, **kwargs):
    return _timed_decision_making(config, mode, **kwargs)

# non-timing tasks

def _spatial_reproduction(config, mode, **kwargs):
    '''
    A pulse with spatial information is presented, after a fixed delay comes another pulse, then the network should indicated the spatial location.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = kwargs['gaussian_center']
        if not hasattr(gaussian_center, '__iter__'):
            gaussian_center = np.array([gaussian_center] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the go cue
    control_on = stim1_off + prod_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration
    # response start time
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # go cue
    trial.add('input', 0, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 1, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.gaussian_center = gaussian_center
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_reproduction(config, mode, **kwargs):
    return _spatial_reproduction(config, mode, **kwargs)


def _spatial_reproduction_broad_tuning(config, mode, **kwargs):
    '''
    The same as spatial reproduction task, except that the Gaussian tuning of input neurons are broadened
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(12., 32+12-12.), batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(12., 32+12-12.), batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = kwargs['gaussian_center']
        if not hasattr(gaussian_center, '__iter__'):
            gaussian_center = np.array([gaussian_center] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the go cue
    control_on = stim1_off + prod_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration
    # response start time
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # go cue
    trial.add('input', 0, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 1, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.gaussian_center = gaussian_center
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_reproduction_broad_tuning(config, mode, **kwargs):
    return _spatial_reproduction_broad_tuning(config, mode, **kwargs)


def _spatial_reproduction_variable_delay(config, mode, **kwargs):
    '''
    The same as spatial reproduction task, except that the delay period is randomly variable
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)

        gaussian_center = rng.choice(np.arange(6., 32-6.), batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gaussian_center = kwargs['gaussian_center']
        if not hasattr(gaussian_center, '__iter__'):
            gaussian_center = np.array([gaussian_center] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the go cue
    control_on = stim1_off + prod_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration
    # response start time
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # go cue
    trial.add('input', 0, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 1, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.gaussian_center = gaussian_center
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_reproduction_variable_delay(config, mode, **kwargs):
    return _spatial_reproduction_variable_delay(config, mode, **kwargs)


def _spatial_comparison(config, mode, **kwargs):
    '''
    Two pulses with spatial locations are presented separately, with a fixed delay in-between. The
    network should indicate which pulse has larger location coordinate.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2 = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2 = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''
        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (gaussian_center1 > gaussian_center2)
    output_target2 = 1. * (gaussian_center1 <= gaussian_center2)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_comparison(config, mode, **kwargs):
    return _spatial_comparison(config, mode, **kwargs)

def _spatial_comparison_broad_tuning(config, mode, **kwargs):
    '''
    The same as spatial comparison task, except that the Gaussian tuning of the inputs are broadened.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(12., 32+12-12.), batch_size)
        gaussian_center2 = rng.choice(np.arange(12., 32+12-12.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(12., 32+12-12.), batch_size)
        gaussian_center2 = rng.choice(np.arange(12., 32+12-12.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''
        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (gaussian_center1 > gaussian_center2)
    output_target2 = 1. * (gaussian_center1 <= gaussian_center2)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_comparison_broad_tuning(config, mode, **kwargs):
    return _spatial_comparison_broad_tuning(config, mode, **kwargs)


def _spatial_comparison_variable_delay(config, mode, **kwargs):
    '''
    The same as spatial comparison task, except that the delay period is randomly variable during training.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(800, 1600, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2 = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(800, 1600, batch_size) / dt).astype(int)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2 = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2[gaussian_center2 >= gaussian_center1] += 1.

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''
        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (gaussian_center1 > gaussian_center2)
    output_target2 = 1. * (gaussian_center1 <= gaussian_center2)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_comparison_variable_delay(config, mode, **kwargs):
    return _spatial_comparison_variable_delay(config, mode, **kwargs)


def _spatial_change_detection(config, mode, **kwargs):
    '''
    Two pulses with spatial locations are successively presented, with a fixed delay in-between. The network should
    respond whether the two spatial locations are the same.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2_unchange = gaussian_center1

        gaussian_center2_change = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2_unchange = gaussian_center1

        gaussian_center2_change = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''

        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (np.abs(gaussian_center1 - gaussian_center2) < 0.5)
    output_target2 = 1. * (np.abs(gaussian_center1 - gaussian_center2) >= 0.5)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_change_detection(config, mode, **kwargs):
    return _spatial_change_detection(config, mode, **kwargs)


def _spatial_change_detection_broad_tuning(config, mode, **kwargs):
    '''
    The same as spatial change detection, except for broad Gaussian tuning.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(12, 32+12-12), batch_size)
        gaussian_center2_unchange = gaussian_center1

        #gaussian_center2_change = rng.choice(np.arange(12., 32+12-12.-1.), batch_size)
        #gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2_change = rng.choice(np.arange(12, 32+12-12-3), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1-1] += 3

        gaussian_center2_change_1 = rng.choice(np.arange(12+2, 32+12-12), batch_size)
        gaussian_center2_change_2 = rng.choice(np.arange(12, 32+12-12-2), batch_size)
        gaussian_center2_change[gaussian_center1==12] = gaussian_center2_change_1[gaussian_center1==12]
        gaussian_center2_change[gaussian_center1==31] = gaussian_center2_change_2[gaussian_center1==31]

        gaussian_center1 = gaussian_center1.astype(np.float64)
        gaussian_center2_unchange = gaussian_center2_unchange.astype(np.float64)
        gaussian_center2_change = gaussian_center2_change.astype(np.float64)

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(12, 32 + 12 - 12), batch_size)
        gaussian_center2_unchange = gaussian_center1

        # gaussian_center2_change = rng.choice(np.arange(12., 32+12-12.-1.), batch_size)
        # gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2_change = rng.choice(np.arange(12, 32 + 12 - 12 - 3), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1 - 1] += 3

        gaussian_center2_change_1 = rng.choice(np.arange(12+2, 32+12-12), batch_size)
        gaussian_center2_change_2 = rng.choice(np.arange(12, 32+12-12-2), batch_size)
        gaussian_center2_change[gaussian_center1==12] = gaussian_center2_change_1[gaussian_center1==12]
        gaussian_center2_change[gaussian_center1==31] = gaussian_center2_change_2[gaussian_center1==31]

        gaussian_center1 = gaussian_center1.astype(np.float64)
        gaussian_center2_unchange = gaussian_center2_unchange.astype(np.float64)
        gaussian_center2_change = gaussian_center2_change.astype(np.float64)

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''

        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (np.abs(gaussian_center1 - gaussian_center2) < 0.5)
    output_target2 = 1. * (np.abs(gaussian_center1 - gaussian_center2) >= 0.5)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_change_detection_broad_tuning(config, mode, **kwargs):
    return _spatial_change_detection_broad_tuning(config, mode, **kwargs)


def _spatial_change_detection_variable_delay(config, mode, **kwargs):
    '''
    The same as spatial change detection, except for randomly variable delay.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60 / dt)
    response_duration = int(300 / dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(800, 1600, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2_unchange = gaussian_center1

        gaussian_center2_change = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        dly_interval = (rng.uniform(800, 1600, batch_size) / dt).astype(int)

        change_choice = rng.choice([0., 1.], batch_size)

        gaussian_center1 = rng.choice(np.arange(6., 32-6.), batch_size)
        gaussian_center2_unchange = gaussian_center1

        gaussian_center2_change = rng.choice(np.arange(6., 32-6.-1.), batch_size)
        gaussian_center2_change[gaussian_center2_change >= gaussian_center1] += 1.

        gaussian_center2 = change_choice * gaussian_center2_change + (1.-change_choice) * gaussian_center2_unchange

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        dly_interval = (rng.uniform(1200, 1200, batch_size) / dt).astype(int)
        '''
        dly_interval = kwargs['dly_interval']
        if not hasattr(dly_interval, '__iter__'):
            dly_interval = np.array([dly_interval] * batch_size)
        dly_interval = (dly_interval / dt).astype(int)
        '''

        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size) / dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size) / dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the second stimulus
    stim2_on = stim1_off + dly_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + pulse_duration
    # response start time
    response_on = stim2_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    # output target
    """
    def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))

    # output
    output_target1 = 1. * (prod_interval1 > prod_interval2)
    output_target2 = 1. * (prod_interval1 <= prod_interval2)

    """
    output_target1 = 1. * (np.abs(gaussian_center1 - gaussian_center2) < 0.5)
    output_target2 = 1. * (np.abs(gaussian_center1 - gaussian_center2) >= 0.5)

    # pulse input
    trial.add('line_gaussian_input', 0, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', trial.n_guassianline, ons=stim2_on, offs=stim2_off, strengths=trial.expand(1.), gaussian_center=gaussian_center2)

    # output
    trial.add('out', 0, ons=stim2_off, offs=xtdim, strengths=output_target1)
    trial.add('out', 1, ons=stim2_off, offs=xtdim, strengths=output_target2)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'go': (stim2_off, response_off)}

    trial.dly_interval = dly_interval
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_change_detection_variable_delay(config, mode, **kwargs):
    return _spatial_change_detection_variable_delay(config, mode, **kwargs)


def _decision_making(config, mode, **kwargs):
    '''
    Two stimuli are presented simultaneously. The network should indicate which stimulus is stronger.
    '''
    dt = config['dt']
    pulse_duration = int(60/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        '''
        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)
        '''
        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval
    response_on = stim1_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # output
    output_target1 = 1. * (strength1 > strength2)
    output_target2 = 1. * (strength1 <= strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'interval': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.strength1 = strength1
    trial.strength2 = strength2

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def decision_making(config, mode, **kwargs):
    return _decision_making(config, mode, **kwargs)


def _decision_making_variable_delay(config, mode, **kwargs):
    '''
    The same as decision making task, except for randomly variable presentation period.
    '''
    dt = config['dt']
    pulse_duration = int(60/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        '''
        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)
        '''
        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval
    response_on = stim1_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # output
    output_target1 = 1. * (strength1 > strength2)
    output_target2 = 1. * (strength1 <= strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'interval': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.strength1 = strength1
    trial.strength2 = strength2

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def decision_making_variable_delay(config, mode, **kwargs):
    return _decision_making_variable_delay(config, mode, **kwargs)

def _ctx_decision_making(config, mode, **kwargs):
    '''
    Two stimuli are presented, a pulse cue is then input to the network at the very end of the presentation.
    The network should respond the index of the stronger or weaker stimulus, depending on the content of the pulse cue.
    '''
    dt = config['dt']
    pulse_duration = int(60/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        choice = rng.choice([0., 1.], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        choice = rng.choice([0., 1.], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        '''
        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)
        '''
        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

        choice = kwargs['choice']
        if not hasattr(choice, '__iter__'):
            choice = np.array([choice] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval
    #control_on = stim1_off
    control_on = stim1_off - pulse_duration
    control_off = control_on + pulse_duration
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    cue_0_input = choice
    cue_1_input = 1. - choice

    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # go cue
    trial.add('input', 2, ons=control_on, offs=control_off, strengths=cue_0_input)
    trial.add('input', 3, ons=control_on, offs=control_off, strengths=cue_1_input)

    # output
    output_target1 = choice * (strength1 > strength2) + (1. - choice) * (strength1 <= strength2)
    output_target2 = choice * (strength1 <= strength2) + (1. - choice) * (strength1 > strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'interval': (stim1_on, stim1_off),
                    'go_cue': (control_on, control_off),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.strength1 = strength1
    trial.strength2 = strength2
    trial.choice = choice

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def ctx_decision_making(config, mode, **kwargs):
    return _ctx_decision_making(config, mode, **kwargs)

def _ctx_decision_making_variable_delay(config, mode, **kwargs):
    '''
    The same as ctx_decision_making, except for randomly variable stimulus presentation period.
    '''
    dt = config['dt']
    pulse_duration = int(60/dt)
    rng = config['rng']
    response_duration = int(300/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        choice = rng.choice([0., 1.], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        prod_interval = (rng.uniform(800, 1600, batch_size)/dt).astype(int)
        gamma_bar = rng.uniform(0.8, 1.2, batch_size)
        c = rng.choice([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08], (batch_size,))

        choice = rng.choice([0., 1.], (batch_size,))

        #c = rng.uniform(-0.06, 0.06, batch_size)

    elif mode == 'test':
        batch_size = kwargs['batch_size']
        '''
        prod_interval = kwargs['prod_interval']
        if not hasattr(prod_interval, '__iter__'):
            prod_interval = np.array([prod_interval] * batch_size)
        prod_interval = (prod_interval / dt).astype(int)
        '''
        prod_interval = (rng.uniform(1200, 1200, batch_size)/dt).astype(int)

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

        choice = kwargs['choice']
        if not hasattr(choice, '__iter__'):
            choice = np.array([choice] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    # the onset time of the first stimulus
    if kwargs['noise_on']:
        stim1_on = (rng.uniform(80, 500, batch_size)/dt).astype(int)
    else:
        stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + prod_interval
    #control_on = stim1_off
    control_on = stim1_off - pulse_duration
    control_off = control_on + pulse_duration
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)

    cue_0_input = choice
    cue_1_input = 1. - choice

    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # go cue
    trial.add('input', 2, ons=control_on, offs=control_off, strengths=cue_0_input)
    trial.add('input', 3, ons=control_on, offs=control_off, strengths=cue_1_input)

    # output
    output_target1 = choice * (strength1 > strength2) + (1. - choice) * (strength1 <= strength2)
    output_target2 = choice * (strength1 <= strength2) + (1. - choice) * (strength1 > strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'interval': (stim1_on, stim1_off),
                    'go_cue': (control_on, control_off),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.strength1 = strength1
    trial.strength2 = strength2
    trial.choice = choice

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def ctx_decision_making_variable_delay(config, mode, **kwargs):
    return _ctx_decision_making_variable_delay(config, mode, **kwargs)


# map string to function
rule_mapping = {
                'interval_production': interval_production,
                'interval_comparison': interval_comparison,
                'timed_spatial_reproduction': timed_spatial_reproduction,
                'timed_spatial_reproduction_broad_tuning': timed_spatial_reproduction_broad_tuning,
                'timed_decision_making': timed_decision_making,
                ####################################

                'spatial_reproduction': spatial_reproduction,
                'spatial_reproduction_broad_tuning': spatial_reproduction_broad_tuning,
                'spatial_reproduction_variable_delay': spatial_reproduction_variable_delay,

                'spatial_comparison': spatial_comparison,
                'spatial_comparison_broad_tuning': spatial_comparison_broad_tuning,
                'spatial_comparison_variable_delay': spatial_comparison_variable_delay,

                'spatial_change_detection': spatial_change_detection,
                'spatial_change_detection_broad_tuning': spatial_change_detection_broad_tuning,
                'spatial_change_detection_variable_delay': spatial_change_detection_variable_delay,
                ####################################

                'decision_making': decision_making,
                'decision_making_variable_delay': decision_making_variable_delay,

                'ctx_decision_making': ctx_decision_making,
                'ctx_decision_making_variable_delay': ctx_decision_making_variable_delay
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


# test the generation of trials
if __name__ == "__main__":
    import seaborn as sns
    from matplotlib import pyplot as plt

    import default

    print(sys.argv)
    print(len(sys.argv))

    min_interval = 100
    max_interval = 500
    num_interval = 3
    train_time_interval = np.arange(min_interval, max_interval + (max_interval - min_interval) / (num_interval - 1),
                                    (max_interval - min_interval) / (num_interval - 1))
    print(train_time_interval)

    rule_name = 'interval_production'

    train_time_interval = np.array([600, 1200])

    hp = default.get_default_hp(rule_name)
    trial = generate_trials(rule_name, hp, 'random_validate', noise_on=True, batch_size=1, train_time_interval=train_time_interval)

    if rule_name == 'interval_production':

        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        # input
        data = trial.x[:, 0, 0]
        plt.plot(range(len(data)), data, color='blue')
        # cue
        data = trial.x[:, 0, 1]
        plt.plot(range(len(data)), data, color='yellow')
        # target output
        data = trial.y[:, 0, 0]
        plt.plot(range(len(data)), data, color='red')

        plt.figure()
        # cost_mask
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()

    elif rule_name == 'decision_making_prod_no_dly':
        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        # input1
        data = trial.x[:, 0, 0]
        plt.plot(range(len(data)), data, color='blue')
        # input2
        data = trial.x[:, 0, 1]
        plt.plot(range(len(data)), data, color='yellow')
        # target output
        data = trial.y[:, 0, 0]
        plt.plot(range(len(data)), data, color='red')
        data = trial.y[:, 0, 1]
        plt.plot(range(len(data)), data, color='pink')

        plt.figure()
        # cost_mask
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()

    elif rule_name == 'timed_decision_making':
        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        # input1
        data = trial.x[:, 0, 0]
        plt.plot(range(len(data)), data, color='blue')
        # input2
        data = trial.x[:, 0, 1]
        plt.plot(range(len(data)), data, color='yellow')
        # cue
        data = trial.x[:, 0, 2]
        plt.plot(range(len(data)), data, color='green')
        # target output
        data = trial.y[:, 0, 0]
        plt.plot(range(len(data)), data, color='red')
        data = trial.y[:, 0, 1]
        plt.plot(range(len(data)), data, color='pink')

        plt.figure()
        # cost_mask
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()

    elif rule_name == 'direction_prod_no_dly':
        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        data = trial.x[:, 0, :]
        sns.heatmap(data)

        plt.figure()
        data = trial.y[:, 0, :]
        sns.heatmap(data)

        plt.figure()
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()

    elif rule_name == 'direction_interval_production':
        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        data = trial.x[:, 0, :]
        sns.heatmap(data)

        plt.figure()
        data = trial.y[:, 0, :]
        sns.heatmap(data)

        plt.figure()
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()
    elif rule_name == 'interval_comparison':
        print(trial.x.shape)
        print(trial.y.shape)
        print(trial.cost_mask.shape)

        plt.figure()
        # input1
        data = trial.x[:, 0, 0]
        plt.plot(range(len(data)), data, color='blue')
        # input2
        data = trial.x[:, 0, 1]
        plt.plot(range(len(data)), data, color='yellow')
        # target output
        data = trial.y[:, 0, 0]
        plt.plot(range(len(data)), data, color='red')
        data = trial.y[:, 0, 1]
        plt.plot(range(len(data)), data, color='pink')

        plt.figure()
        # cost_mask
        data = trial.cost_mask[:, 0, 0]
        plt.plot(range(len(data)), data, color='black')
        plt.show()
