import sys
import errno
import numpy as np

import os

from core.data_plot import timed_spatial_reproduction_decoding_lib


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

mkdir_p("./figure/fig4")

fig=timed_spatial_reproduction_decoding_lib.time_flow_decoding_generalization_across_space(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], epoch='go', decoder_type=1, transient_period=10, go_scaling=False, redo=False)
fig.savefig("./figure/fig4/time_flow_generalization_across_space_vs_angle_and_mix_var_decoding_type1.pdf")

fig=timed_spatial_reproduction_decoding_lib.time_flow_decoding_generalization_across_space(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], epoch='go', decoder_type=2, transient_period=10,go_scaling=False, redo=False)
fig.savefig("./figure/fig4/time_flow_generalization_across_space_vs_angle_and_mix_var_decoding_type2.pdf")

total_result_decoding1, total_result_mix_var, total_result_angle = timed_spatial_reproduction_decoding_lib.time_flow_decoding_generalization_across_space(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], epoch='go', decoder_type=1, transient_period=10, fig_on=False,go_scaling=False, redo=False)
total_result_decoding2, total_result_mix_var, total_result_angle = timed_spatial_reproduction_decoding_lib.time_flow_decoding_generalization_across_space(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], epoch='go', decoder_type=2, transient_period=10, fig_on=False,go_scaling=False, redo=False)
fig1 = timed_spatial_reproduction_decoding_lib.direct_correlation_by_decimation(total_result_angle, total_result_mix_var, total_result_decoding1, decim_num=2)
fig2 = timed_spatial_reproduction_decoding_lib.direct_correlation_by_decimation(total_result_angle, total_result_mix_var, total_result_decoding2, decim_num=2)
fig1.savefig("./figure/fig4/direct_correlation_by_decimation_decoding_type1.pdf")
fig2.savefig("./figure/fig4/direct_correlation_by_decimation_decoding_type2.pdf")

fig=timed_spatial_reproduction_decoding_lib.time_flow_generalization(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], epoch='go')
fig.savefig("./figure/fig4/time_flow_generalization_production.pdf")