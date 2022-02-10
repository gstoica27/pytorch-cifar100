#!/bin/bash

tmux new-session -d -s CSAM300 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM301 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM302 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [4, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [4, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM303 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [5, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM304 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [4, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [4, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM305 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [5, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM306 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[4, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[4, 1, 3], [5, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM307 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [4, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [4, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM308 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [5, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM309 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [4, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[3, 1, 3], [4, 1, 3], [5, 1, 3]]" --stride 1'

tmux new-session -d -s CSAM310 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [4, 1, 3], [5, 1, 3]]" --stride 1'

echo 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate p3 && srun -p overcap -A overcap -t 48:00:00 --gres gpu:1 -c 6 python train.py -net "resnet18"  --approach_name "3" --pos_emb_dim 0 --injection_info "[[2, 1, 3], [3, 1, 3], [4, 1, 3], [5, 1, 3]]" --stride 1'