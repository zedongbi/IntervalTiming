import queue
import threading
import os
import sys

#the number of threads that run simultaneously
num_worker_threads = 2


def train_model(rule_name, w2_reg, r2_reg, index):
    run_cmd = 'python core/cluster_training_unit.py '+rule_name+' '+str(w2_reg)+' '+str(r2_reg)+' '+str(index)
    os.system(run_cmd)


#####################################################################
parameter_list = list()


def set_parameter_list(rule_name, indexes=list(range(0, 16)), w2_reg_list=[0], r2_reg_list=[0]):
    for w2_reg in w2_reg_list:
        for r2_reg in r2_reg_list:
            for index in indexes:
                parameter_list.append((rule_name, w2_reg, r2_reg, index))

'''
def set_parameter_main():
    thread_index = int(sys.argv[1])
    if thread_index == 0:
        set_parameter_list('interval_production', indexes=[0])
        set_parameter_list('interval_comparison', indexes=[0])
    if thread_index == 1:
        set_parameter_list('timed_spatial_reproduction', indexes=[0])
        set_parameter_list('timed_spatial_reproduction_broad_tuning', indexes=[0])
    if thread_index == 2:
        set_parameter_list('timed_decision_making', indexes=[0])
        set_parameter_list('spatial_reproduction', indexes=[0])
    if thread_index == 3:
        set_parameter_list('spatial_reproduction_broad_tuning', indexes=[0])
        set_parameter_list('spatial_reproduction_variable_delay', indexes=[0])
    if thread_index == 4:
        set_parameter_list('spatial_comparison_broad_tuning', indexes=[0])
        set_parameter_list('spatial_comparison_variable_delay', indexes=[0])
    if thread_index == 5:
        set_parameter_list('spatial_change_detection', indexes=[0])
        set_parameter_list('spatial_change_detection_broad_tuning', indexes=[0])
    if thread_index == 6:
        set_parameter_list('spatial_change_detection_variable_delay', indexes=[0])
        set_parameter_list('decision_making', indexes=[0])
    if thread_index == 7:
        set_parameter_list('decision_making_variable_delay', indexes=[0])
        set_parameter_list('ctx_decision_making', indexes=[0])
    if thread_index == 8:
        set_parameter_list('ctx_decision_making_variable_delay', indexes=[0])
        set_parameter_list('spatial_comparison', indexes=[0])

set_parameter_main()
'''
# to train the network on a task, set the argument of the functio to be the same of the task
set_parameter_list('interval_production', indexes=[0])
set_parameter_list('interval_comparison', indexes=[0])

set_parameter_list('timed_spatial_reproduction', indexes=[0])
set_parameter_list('timed_spatial_reproduction_broad_tuning', indexes=[0])

set_parameter_list('timed_decision_making', indexes=[0])

set_parameter_list('spatial_reproduction', indexes=[0])
set_parameter_list('spatial_reproduction_broad_tuning', indexes=[0])
set_parameter_list('spatial_reproduction_variable_delay', indexes=[0])

set_parameter_list('spatial_comparison', indexes=[0])
set_parameter_list('spatial_comparison_broad_tuning', indexes=[0])
set_parameter_list('spatial_comparison_variable_delay', indexes=[0])

set_parameter_list('spatial_change_detection', indexes=[0])

set_parameter_list('spatial_change_detection_broad_tuning', indexes=[0])
set_parameter_list('spatial_change_detection_variable_delay', indexes=[0])

set_parameter_list('decision_making', indexes=[0])
set_parameter_list('decision_making_variable_delay', indexes=[0])
set_parameter_list('ctx_decision_making', indexes=[0])
set_parameter_list('ctx_decision_making_variable_delay', indexes=[0])


#####################################################################

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        train_model(*item)
        q.task_done()


q = queue.Queue()

threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)


for item in parameter_list:
    q.put(item)


# block until all tasks are done
q.join()

# stop workers
for i in range(num_worker_threads):
    q.put(None)
for t in threads:
    t.join()
