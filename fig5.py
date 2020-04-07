import errno
import numpy as np

import os

from core.data_plot import interval_production_lib
from core.data_plot import timed_spatial_reproduction_lib
from core.data_plot import timed_decision_making_lib


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

mkdir_p("./figure/fig5")

fig = interval_production_lib.activity_peak_order_plot('w2_0.0_r2_0.0/model_1/finalResult', epoch='interval', func_activity_threshold=2)
fig.savefig("./figure/fig5/sequence_perception_epoch.pdf")

fig=interval_production_lib.connection_peak_order_plot_batch(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(16)], epoch='interval', func_activity_threshold=2)
fig.savefig("./figure/fig5/peak_order_weight_perception_epoch.pdf")

fig=timed_spatial_reproduction_lib.weight_connection_with_time(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(30)], time_points=np.arange(300, 1000, 100),func_relevance_threshold=2)
fig.savefig("./figure/fig5/frequency_prod_dly_weight_connection_with_time_perception_epoch.pdf")

fig=timed_decision_making_lib.weight_connection_with_time(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], time_points=np.arange(300, 1000, 100),func_relevance_threshold=2.)
fig.savefig("./figure/fig5/weight_connection_with_time_perception_epoch.pdf")