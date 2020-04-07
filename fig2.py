import errno
import numpy as np
import os


from core.data_plot import interval_production_lib


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

mkdir_p("./figure/fig2")

fig = interval_production_lib.neuron_activity_plot_three_epochs('w2_0.0_r2_0.0/model_3/finalResult')
fig.savefig("./figure/fig2/neuron_activity_plot_three_epochs.pdf")

fig1, fig2 = interval_production_lib.PCA_plot('w2_0.0_r2_0.0/model_0/finalResult', 'interval', noise_on=True)
fig1.savefig("./figure/fig2/PCA_plot_perception_1.pdf")
fig1.savefig("./figure/fig2/PCA_plot_perception_2.pdf")

fig=interval_production_lib.neuron_activity_example_plot('w2_0.0_r2_0.0/model_4/finalResult', epoch='interval', noise_on=True)
fig.savefig("./figure/fig2/neuron_activity_example_plot.pdf")

fig=interval_production_lib.explained_var_interval_r2(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], 'interval', noise_on=True)
fig.savefig("./figure/fig2/sequence_r2_perception_epoch.pdf")

fig=interval_production_lib.PCA_velocity_plot('w2_0.0_r2_0.0/model_0/finalResult', 'delay')
fig.savefig("./figure/fig2/PCA_velocity_plot.pdf")

fig=interval_production_lib.velocity_during_delay_plot_batch(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], 'delay', prod_intervals=np.array([600, 1200]))
fig.savefig("./figure/fig2/velocity_during_delay_plot_batch.pdf")

fig=interval_production_lib.end_delay_dim_plot_batch(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], 'delay', prod_intervals=np.arange(600, 1300, 100))
fig.savefig("./figure/fig2/end_dim_delay_epoch.pdf")

fig=interval_production_lib.end_delay_topology(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)], 'delay', prod_intervals=np.arange(600, 1300, 100))
fig.savefig("./figure/fig2/end_delay_topology.pdf")

fig=interval_production_lib.adjacent_trajectory_distance_delay_epoch(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in list(range(0,16))])
fig.savefig("./figure/fig2/adjacent_trajectory_distance_delay_epoch.pdf")

fig=interval_production_lib.neuron_tuning_end_of_delay_grid_plot('w2_0.0_r2_0.0/model_0/finalResult',func_relevance_threshold=2)
fig.savefig("./figure/fig2/neuron_tuning_end_of_delay_grid_plot.pdf")

fig=interval_production_lib.neuron_type_end_of_delay(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in range(16)],func_relevance_threshold=2)
fig.savefig("./figure/fig2/neuron_type_end_of_delay.pdf")

fig1, fig2 = interval_production_lib.PCA_plot('w2_0.0_r2_0.0/model_2/finalResult', 'go')
fig1.savefig("./figure/fig2/pca3_production_epoch.pdf")
fig2.savefig("./figure/fig2/pca3_colorbar_production_epoch.pdf")

fig = interval_production_lib.neuron_activity_example_plot_no_scaling_vs_scaling('w2_0.0_r2_0.0/model_3/finalResult', epoch='go')
fig.savefig("./figure/fig2/example_activity_production_epoch_no_scaling_vs_scaling.pdf")

fig=interval_production_lib.production_epoch_explained_variance_plot_batch(['w2_0.0_r2_0.0/model_'+str(i)+'/finalResult' for i in range(16)], 'go')
fig.savefig("./figure/fig2/scaling_index_production_epoch.pdf")

fig=interval_production_lib.velocity_in_scaling_space(['w2_0.0_r2_0.0/model_'+str(x)+'/finalResult' for x in list(range(0,16))], epoch='go',velocity_dim_n=3)
fig.savefig("./figure/fig2/velocity_production_epoch.pdf")