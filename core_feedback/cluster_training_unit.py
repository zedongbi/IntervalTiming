import os
import sys

sys.path.append(os.getcwd())

from core_feedback import task
from core_feedback import network
from core_feedback import train
from core_feedback import tools
from core_feedback import default


def train_model(rule_name, w2_reg, r2_reg, signal2Strength, index):

    serial_idx = os.path.join('w2_'+str(w2_reg)+'_r2_'+str(r2_reg)+'_signal2Strength_'+str(signal2Strength), 'model_'+str(index))

    local_folder_name = os.path.join('./core_feedback/model', rule_name, serial_idx)

    kwargs = dict()
    kwargs['signal2_strength'] = signal2Strength

    while True:
        hp = default.get_default_hp(rule_name)
        hp['l2_firing_rate'] = r2_reg
        hp['l2_weight'] = w2_reg

        trainerObj = train.Trainer(model_dir=local_folder_name, rule_name=rule_name, hp=hp, **kwargs)

        stat = trainerObj.train(max_samples=1e7, display_step=200)
        if stat is 'OK':
            break
        else:
            run_cmd = 'rm -r ' + local_folder_name
            os.system(run_cmd)


if __name__ == "__main__":
    rule_name = sys.argv[1]
    w2_reg = float(sys.argv[2])
    r2_reg = float(sys.argv[3])
    signal2Strength = float(sys.argv[4])
    index = int(sys.argv[5])

    print(rule_name, w2_reg, r2_reg, signal2Strength, index)

    train_model(rule_name, w2_reg, r2_reg, signal2Strength, index)
