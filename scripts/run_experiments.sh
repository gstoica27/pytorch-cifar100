#!/bin/bash

tmux new-session -d -s CSAM0 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "1" --pos_emb_dim 0 --injection_info "[[1, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "1" --pos_emb_dim 0 --injection_info "[[1, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM1 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "1" --pos_emb_dim 0 --injection_info "[[1, 1, 3], [2, 1, 3], [3, 1, 3], [4, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "1" --pos_emb_dim 0 --injection_info "[[1, 1, 3], [2, 1, 3], [3, 1, 3], [4, 1, 3]]" --stride 1'