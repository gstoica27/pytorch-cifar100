import os

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
approach_names = ['self_attention']
filter_sizes = [1]
strides = [1]
stackings = [1]
injection_points = [
    [3],
]
positional_encodings = [0]
residuals = ['True']

# Aggregate specifications
experiment_list = []
for repetition in range(num_repetitions):
    for approach_name in approach_names:
        for stacking in stackings:
            for injection_point in injection_points:
                for pos_enc in positional_encodings:
                    for residual in residuals:
                        for filter_size in filter_sizes:
                            for stride in strides:
                                experiment_list.append(
                                    ["resnet", approach_name, filter_size, stride, injection_point, stacking, pos_enc, residual, repetition]
                                )

def generate_command(experiment_config, env_name):
    resume_arg = " -resume" if do_resume else ""
    a40_constraint_arg = " --constraint=a40" if require_a40 else ""

    injection_info = [
        [i, experiment_config[5], experiment_config[2]] for i in experiment_config[4]
    ]

    residual_connection_arg = " --use_residual_connection" if experiment_config[7] == 'True' else ""

    out = (
        f"'source /srv/share4/thearn6/miniconda3/etc/profile.d/conda.sh && conda activate {env_name} && "
        f"srun -p overcap -A overcap{a40_constraint_arg} -t 48:00:00"
        + f' --gres gpu:1 -c 6 python train.py{resume_arg} -net "resnet18" '
        + f' --approach_name "{experiment_config[1]}"'
        + f' --suffix "{experiment_config[8]}"'
        + f" --pos_emb_dim {experiment_config[6]}"
        + f' --injection_info "{injection_info}"'
        + f" --stride {experiment_config[3]}"
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

        tmux_prefix = f"tmux new-session -d -s CSAM{approach_name.replace('_','')}{tmux_ind} "
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
