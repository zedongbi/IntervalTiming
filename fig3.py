import errno
import numpy as np

import os

from core.data_plot import timed_spatial_reproduction_lib


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

mkdir_p("./figure/fig3")

fig=timed_spatial_reproduction_lib.firing_rate_compare(serial_idx='w2_0.0_r2_0.0/model_2/finalResult', mode='draw')
fig.savefig("./figure/fig3/firing_rate_compare.pdf")

fig=timed_spatial_reproduction_lib.PCA_2d_plot_with_time_stim_direction(serial_idx='w2_0.0_r2_0.0/model_3/finalResult', epoch='interval', noise_on=False)
fig.savefig("./figure/fig3/PCA_2d_plot_with_time_stim_direction.pdf")

fig1=timed_spatial_reproduction_lib.orthogonality_perception(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)])
fig1.savefig("./figure/fig3/orthogonality_perception.pdf")

fig1=timed_spatial_reproduction_lib.portion_of_variance_perception(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)])
fig1.savefig("./figure/fig3/portion_of_variance_perception.pdf")

fig=timed_spatial_reproduction_lib.orthogonality_delay(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)])
fig.savefig("./figure/fig3/orthogonality_delay.pdf")

fig=timed_spatial_reproduction_lib.portion_of_variance_delay(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)])
fig.savefig("./figure/fig3/portion_of_variance_delay.pdf")

fig1, fig2, fig3=timed_spatial_reproduction_lib.orthogonality_production(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)])
fig1.savefig("./figure/fig3/orthogonality_production1.pdf")
fig2.savefig("./figure/fig3/orthogonality_production2.pdf")
fig3.savefig("./figure/fig3/orthogonality_production3.pdf")

fig=timed_spatial_reproduction_lib.portion_of_variance_production(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(0,30)], mode='calculate')
fig.savefig("./figure/fig3/portion_of_variance_production.pdf")