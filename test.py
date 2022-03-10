#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
from email.errors import FirstHeaderLineIsContinuationDefect
from email.policy import default

from matplotlib import pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from conf import settings
from utils import get_network, get_test_dataloader, read_yaml
from models.csam import ConvAttnWrapper
import pdb
import scipy.stats as st
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    # parser.add_argument('-variant_name', type=str, required=True, help='approach variant')
    # parser.add_argument('-position_encoding_dim', type=int, default=10, help='positional encoding dimension')
    # parser.add_argument('-variant_loc', type=int, default=5, help='location where to add module')
    # parser.add_argument('-softmax_temp', type=int, default=1, help='cosine similarity softmax temp')
    parser.add_argument('--checkpoints_dir', type=str, default='/srv/share4/gstoica3/checkpoints/resnet18', help='directory of model to load')
    parser.add_argument('--approach_name', type=str)
    parser.add_argument('--subdir_file', type=str)

    args = parser.parse_args()
    subdir2results = {}
    backbone = get_network(args)
    
    subdirs = []
    with open(args.subdir_file, 'r') as handle:
        for line in handle:
            subdirs.append(line.strip().replace('./', ''))

    for subdir in subdirs:
        try:
            model_dir = os.path.join(args.checkpoints_dir, args.approach_name, subdir)
            config_path = os.path.join(model_dir, 'convattn.yaml')
            variant_config = read_yaml(config_path)
            model = ConvAttnWrapper(backbone=backbone, variant_kwargs=variant_config).to('cuda:0')
        except:
            continue

        cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            #settings.CIFAR100_PATH,
            num_workers=4,
            batch_size=args.b,
        )
        # pdb.set_trace()
        potential_checkpoints = sorted([checkpoint for checkpoint in os.listdir(model_dir) if 'best' in checkpoint])
        best_checkpoint_path = os.path.join(model_dir, potential_checkpoints[-1])

        weights = torch.load(best_checkpoint_path)
        model_params = {i[0]:i[1] for i in model.named_parameters()}
        # valid_weights = {k:v for k,v in weights.items() if k in named_parameters}
        # import pdb; pdb.set_trace()
        model.load_state_dict(weights)
        print(model)
        model.eval()

        correct_1 = 0.0
        correct_5 = 0.0
        total = 0

        with torch.no_grad():
            for n_iter, (image, label) in tqdm(enumerate(cifar100_test_loader)):
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

                # if args.gpu:
                image = image.cuda()
                label = label.cuda()
                    # print('GPU INFO.....')
                    # print(torch.cuda.memory_summary(), end='')


                output = model(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()

        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        
        top1_error = 1 - correct_1 / len(cifar100_test_loader.dataset)
        top5_error = 1 - correct_5 / len(cifar100_test_loader.dataset)
        
        print()
        print("Top 1 err: ", top1_error)
        print("Top 5 err: ", top5_error)
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
        print('From Checkpoint: {}'.format(best_checkpoint_path))
        subdir2results[subdir] = {'top1err':top1_error.detach().cpu().numpy(), 'top5err': top5_error.detach().cpu().numpy()}
    for subdir, results in subdir2results.items():
        print('{} | Top1 Error: {} | Top5 Error: {}'.format(
            subdir.split('/')[0].replace('CSAM_Approach', ''), results['top1err'], results['top5err']
            )
        )
    data_top1 = [results['top1err'] for results in subdir2results.values()]
    data_top5 = [results['top5err'] for results in subdir2results.values()]
    
    interval_top1 = st.t.interval(
        alpha=0.95, 
        df=len(data_top1)-1, 
        loc=np.mean(data_top1), 
        scale=st.sem(data_top1)
    )

    interval_top5 = st.t.interval(
        alpha=0.95, 
        df=len(data_top5)-1, 
        loc=np.mean(data_top5), 
        scale=st.sem(data_top5)
    )

    print('Confidence Intervals | Top 1: {} ± {} | Top 5: {} ± {}'.format(
        np.mean(data_top1), np.mean(data_top1) - interval_top1[0],
        np.mean(data_top5), np.mean(data_top5) - interval_top5[0]
    ))

