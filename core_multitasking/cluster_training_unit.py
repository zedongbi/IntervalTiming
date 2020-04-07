import os
import sys

sys.path.append(os.getcwd())

from core_multitasking import task
from core_multitasking import network
from core_multitasking import train
from core_multitasking import tools
from core_multitasking import default


def train_model(rule_group, w2_reg, r2_reg, index, **kwargs):

    if rule_group == 'decision_making':
        rule_name_list = ['timed_decision_making', 'decision_making']
    elif rule_group == 'space':
        rule_name_list = ['timed_spatial_reproduction', 'spatial_reproduction']
    elif rule_group == 'space_broad_tuning':
        rule_name_list = ['timed_spatial_reproduction_broad_tuning', 'spatial_reproduction_broad_tuning']
    else:
        raise ValueError('Unknown rule_group')

    serial_idx = os.path.join('w2_'+str(w2_reg)+'_r2_'+str(r2_reg), 'model_'+str(index))

    local_folder_name = os.path.join('./core_multitasking/model', rule_group, serial_idx)

    while True:
        hp = default.get_default_hp(rule_name_list[0])
        hp['l2_firing_rate'] = r2_reg
        hp['l2_weight'] = w2_reg

        trainerObj = train.Trainer(model_dir=local_folder_name, rule_name_list=rule_name_list, hp=hp, **kwargs)
        stat = trainerObj.train(max_samples=1e7, display_step=200)
        if stat is 'OK':
            break
        else:
            run_cmd = 'rm -r ' + local_folder_name
            os.system(run_cmd)


if __name__ == "__main__":
    rule_group = sys.argv[1]
    w2_reg = float(sys.argv[2])
    r2_reg = float(sys.argv[3])
    index = int(sys.argv[4])

    print(rule_group, w2_reg, r2_reg, index)

    train_model(rule_group, w2_reg, r2_reg, index)
