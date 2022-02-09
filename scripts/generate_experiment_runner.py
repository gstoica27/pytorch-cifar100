import os

"""
Use this file to bulk run experiments!

1. Add experiments to the experiment list
2. Change the conda env name in the main function
3. Change the offset value in the main function (to something high) if there are straggling tmux sessions from your previous batch run
4. Run "python scripts/generate_experiment_runner.py"
5. This should generate a scripts/run_experiments.sh, which you can execute for your batch run.
"""

# experiment_list = [
#     # backbone, approach, filter size, stride, location, stacking, position encoding, use_residual_connection
#     ["resnet", "2", 3, 1, [5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [4], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 3], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 4], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [3, 4], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [3, 5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [4, 5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 3, 4], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 3, 5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [3, 4, 5], 1, 10, "False"],
#     ["resnet", "2", 3, 1, [2, 3, 4, 5], 1, 10, "False"],
# ]

# experiment_list = [
#     # backbone, approach, filter size, stride, location, stacking, position encoding, use_residual_connection
#     ["resnet", "3", 3, 1, [3], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,3], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,4], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,5], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [3,4], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [3,5], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [4,5], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,3,4], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,3,5], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [3,4,5], 1, 0, "False"],
#     ["resnet", "3", 3, 1, [2,3,4,5], 1, 0, "False"],
# ]




def generate_command(experiment_config, env_name):

    injection_info = [
        [i, experiment_config[5], experiment_config[2]] for i in experiment_config[4]
    ]

    residual_connection_arg = " --use_residual_connection" if experiment_config[7] == 'True' else ""

    out = (
        f"'source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && "
        "srun -p overcap -A overcap -t 48:00:00"
        + ' --gres gpu:1 -c 6 python train.py -net "resnet18" '
        + f' --approach_name "{experiment_config[1]}"'
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


def generate_bash_executable(env_name="p3", offset=0):

    # Set offset value incase tmux sessions have a chance of having duplicate names
    executable = """#!/bin/bash"""

    for i, experiment in enumerate(experiment_list):

        tmux_ind = offset + i

        executable += "\n\n"

        tmux_prefix = f"tmux new-session -d -s CSAM{tmux_ind} "
        executable += tmux_prefix

        command = generate_command(experiment, env_name)
        executable += command

        executable += "\n\n"

        executable += f"echo {command}"

    with open("scripts/run_experiments.sh", "w") as f:
        f.write(executable)

    make_executable("scripts/run_experiments.sh")


if __name__ == "__main__":
    generate_bash_executable(env_name="p3", offset=300)
