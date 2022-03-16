from datetime import date
import time
import numpy as np
import itertools
import os

"""
Use this file to bulk run experiments!

1. Add experiments to the experiment list
2. Change the conda env name in the main function
3. Change the offset value in the main function (to something high) if there are straggling tmux sessions from your previous batch run
4. Run "python scripts/generate_experiment_runner.py"
5. This should generate a scripts/run_experiments.sh, which you can execute for your batch run.
"""

def get_seed(repetition_num):
    defaults = [17, 0 , 2019, 2022, 1776]
    return defaults[repetition_num] if repetition_num < len(defaults) else repetition_num + np.max(defaults)

# Logistical Options
do_resume = False
require_a40 = False
num_repetitions = 1
on_overcap = True

# Specify configs
approach_names = ['3_unmasked_sv']
filter_sizes = [3]
strides = [1]
stackings = [1]
injection_points = [
    [3],
]
positional_encodings = [0]
residuals = ['False']
forget_gate_nonlinearities = ['sigmoid']
similarity_metrics = ['cosine_similarity']
seeds = [get_seed(i) for i in range(num_repetitions)]

experiment_list = itertools.product(
    approach_names,
    filter_sizes, 
    strides,
    stackings,
    injection_points,
    positional_encodings,
    residuals,
    forget_gate_nonlinearities,
    similarity_metrics,
    seeds
)

indices = {
    'approach_name': 0,
    'filter_size': 1,
    'stride': 2,
    'stacking': 3,
    'injection_point': 4,
    'positional_encoding': 5,
    'residual': 6,
    'forget_gate_nonlinearity': 7,
    'similarity_metric': 8,
    'seed': 9
}

def get_injection_info(config):
    return [
        [i, config[indices['stacking']], config[indices['filter_size']]] for i in config[indices['injection_point']]
    ]

def log_path(config):
    script_dir = os.path.dirname(os.path.realpath(__file__)) 
    log_root_dir = os.path.join(os.path.dirname(script_dir), "logs/experiment_output")
    log_dir = os.path.join(log_root_dir, config[indices['approach_name']], date.today().isoformat())

    formatted_injection_info = ''.join([str(tuple(info)) for info in get_injection_info(config)])
    formatted_injection_info = formatted_injection_info.replace(' ', '').replace('(', '[').replace(')', ']')
    model_name = 'PosEmb{}_AfterConv{}_Stride{}_Residual{}_Forget{}_SimMetr{}_Seed{}'.format(
         config[indices['positional_encoding']],
         formatted_injection_info,
         config[indices['stride']],
         config[indices['residual']],
         config[indices['forget_gate_nonlinearity']],
         config[indices['similarity_metric']],
         config[indices['seed']],
    )
    local_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f'{local_time}_{model_name}.txt'

    return os.path.join(log_dir, file_name)


def generate_command(config, env_name):
    resume_arg = " -resume" if do_resume else ""
    a40_constraint_arg = " --constraint=a40" if require_a40 else ""
    request_params = 'overcap -A overcap' if on_overcap else 'short'

    residual_connection_arg = " --use_residual_connection" if config[indices['residual']] == 'True' else ""

    out = (
        f''''source /srv/share4/thearn6/miniconda3/etc/profile.d/conda.sh && conda activate {env_name} && mkdir -p "`dirname {log_path(config)}`" &&'''
        f''' srun -p {request_params}{a40_constraint_arg} -t 48:00:00'''
        + f''' --gres gpu:1 -c 6'''
        + f''' python train.py{resume_arg} -net "resnet18" '''
        + f''' --approach_name "{config[indices['approach_name']]}"'''
        #+ f''' --suffix "{config[indices['repetition']]}"'''
        + f''' --pos_emb_dim {config[indices['positional_encoding']]}'''
        + f''' --injection_info "{get_injection_info(config)}"'''
        + f''' --stride "{config[indices['stride']]}"'''
        + f''' --forget_gate_nonlinearity "{config[indices['forget_gate_nonlinearity']]}"'''
        + f''' --similarity_metric "{config[indices['similarity_metric']]}"'''
        + f''' --seed "{config[indices['seed']]}"'''
        + residual_connection_arg
        + f''' 2>&1 | tee "{log_path(config)}"'''
        + "'"
    )

    return out

def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def generate_bash_executable(env_name="csam", offset=0):

    # Set offset value incase tmux sessions have a chance of having duplicate names
    executable = """#!/bin/bash"""

    for i, experiment in enumerate(experiment_list):

        tmux_ind = offset + i

        executable += "\n\n"

        tmux_prefix = f"tmux new-session -d -s CSAM{experiment[indices['approach_name']].replace('_','')}{tmux_ind} "
        executable += tmux_prefix

        command = generate_command(experiment, env_name)
        executable += command

        executable += "\n\n"

        executable += f"echo {command}"

    with open("scripts/run_experiments.sh", "w") as f:
        f.write(executable)

    make_executable("scripts/run_experiments.sh")


if __name__ == "__main__":
    generate_bash_executable(env_name="csam", offset=200)
