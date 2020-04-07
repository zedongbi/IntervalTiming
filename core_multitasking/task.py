"""Collections of tasks."""

from __future__ import division
import numpy as np
import math


# all rules
rules_list = list() #['direction_prod_dly', 'direction_no_timing']
#rules_list = ['timed_decision_making', 'decision_making']


def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    # Store indices of rules
    rule_index_map = dict()
    for ind, rule in enumerate(rules_list):
        rule_index_map[rule] = ind

    return rule_index_map[rule] + config['rule_start_idx']


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
                or config['rule_name'] == 'spatial_reproduction_broad_tuning':
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

    # In this version, I suppose that there is only a single stimulus channel and a single output channel
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

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """
        Add rule input.

        on, off and strength are supposed to be scalar values

        """

        self.x[on:off, :, get_rule_index(rule, self.config)] = strength

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x


def _timed_spatial_reproduction(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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


def _spatial_reproduction(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 2, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

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


def _timed_spatial_reproduction_broad_tuning(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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


def _spatial_reproduction_broad_tuning(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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
    trial.add('input', 1, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 2, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

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



def _timed_decision_making(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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


def _decision_making(config, mode, **kwargs):
    '''
    Two pulses are successively input to the network, with interval
    dt_stim. After dt_delay, a 'go' cue is input to the network,
    then the network is required to output a movement at the
    time dt_stim after the 'go' cue.

    The control input is shown between (0, T)
    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stimu2_on, stim2_off)
    The response epoch is between (stim2_off, T)

    :param mode: the mode of generating. Options: 'random', 'test', 'psychometric'
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='test')
    :param param: a dictionary of parameters (required for mode=='psychometric')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
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
    '''
    timed_decision_making
    # input 0
    trial.add('input', 0, ons=stim1_on, offs=stim1_off, strengths=gamma_bar+c)
    # input 1
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=gamma_bar-c)
    # cue
    trial.add('input', 2, ons=cue_on, offs=cue_off, strengths=trial.expand(1.))
    '''
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


# map string to function
rule_mapping = {
                'timed_decision_making': timed_decision_making,
                'decision_making': decision_making,
                'timed_spatial_reproduction': timed_spatial_reproduction,
                'spatial_reproduction': spatial_reproduction,
                'timed_spatial_reproduction_broad_tuning': timed_spatial_reproduction_broad_tuning,
                'spatial_reproduction_broad_tuning': spatial_reproduction_broad_tuning
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

    global rules_list

    if rule == 'timed_decision_making' or rule == 'decision_making':
        rules_list = ['timed_decision_making', 'decision_making']
    elif rule == 'timed_spatial_reproduction' or rule == 'spatial_reproduction':
        rules_list = ['timed_spatial_reproduction', 'spatial_reproduction']
    elif rule == 'timed_spatial_reproduction_broad_tuning' or rule == 'spatial_reproduction_broad_tuning':
        rules_list = ['timed_spatial_reproduction_broad_tuning', 'spatial_reproduction_broad_tuning']

    trial.add_rule(rule, on=None, off=None, strength=1)

    if noise_on:
        trial.add_x_noise()

    return trial
