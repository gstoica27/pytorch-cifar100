import os
import itertools

"""
Use this file to bulk run experiments!

1. Add experiments to the experiment list
2. Change the conda env name in the main function
3. Change the offset value in the main function (to something high) if there are straggling tmux sessions from your previous batch run
4. Run "python scripts/generate_experiment_runner.py"
5. This should generate a scripts/run_experiments.sh, which you can execute for your batch run.
"""

# Logistical Options
do_resume = False
require_a40 = False
num_repetitions = 1

# Specify configs
approach_names = ['3_unmasked']
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

experiment_list = itertools.product(
    approach_names,
    filter_sizes, 
    strides,
    stackings,
    injection_points,
    positional_encodings,
    residuals,
    range(num_repetitions),
    forget_gate_nonlinearities,
    similarity_metrics
)

indices = {
    'approach_name': 0,
    'filter_size': 1,
    'stride': 2,
    'stacking': 3,
    'injection_point': 4,
    'positional_encoding': 5,
    'residual': 6,
    'repetition': 7,
    'forget_gate_nonlinearity': 8,
    'similarity_metric': 9,
}

def generate_command(config, env_name):
    resume_arg = " -resume" if do_resume else ""
    a40_constraint_arg = " --constraint=a40" if require_a40 else ""

    injection_info = [
        [i, config[indices['stacking']], config[indices['filter_size']]] for i in config[indices['injection_point']]
    ]

    residual_connection_arg = " --use_residual_connection" if config[indices['residual']] == 'True' else ""

    out = (
        f"'source /srv/share4/thearn6/miniconda3/etc/profile.d/conda.sh && conda activate {env_name} && "
        f"srun -p overcap -A overcap{a40_constraint_arg} -t 48:00:00"
        + f''' --gres gpu:1 -c 6 python train.py{resume_arg} -net "resnet18" '''
        + f''' --approach_name "{config[indices['approach_name']]}"'''
        + f''' --suffix "{config[indices['repetition']]}"'''
        + f''' --pos_emb_dim {config[indices['positional_encoding']]}'''
        + f''' --injection_info "{injection_info}"'''
        + f''' --stride "{config[indices['stride']]}"'''
        + f''' --forget_gate_nonlinearity "{config[indices['forget_gate_nonlinearity']]}"'''
        + f''' --similarity_metric "{config[indices['similarity_metric']]}"'''
        + residual_connection_arg
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
    generate_bash_executable(env_name="csam", offset=300)
