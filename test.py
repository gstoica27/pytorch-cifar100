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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    # parser.add_argument('-variant_name', type=str, required=True, help='approach variant')
    # parser.add_argument('-position_encoding_dim', type=int, default=10, help='positional encoding dimension')
    # parser.add_argument('-variant_loc', type=int, default=5, help='location where to add module')
    # parser.add_argument('-softmax_temp', type=int, default=1, help='cosine similarity softmax temp')
    parser.add_argument('--model_dir', type=str, help='directory of model to load')

    args = parser.parse_args()
    
    backbone = get_network(args)

    config_path = os.path.join(args.model_dir, 'convattn.yaml')
    variant_config = read_yaml(config_path)
    model = ConvAttnWrapper(backbone=backbone, variant_kwargs=variant_config).to('cuda:0')

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )
    # pdb.set_trace()
    potential_checkpoints = sorted([checkpoint for checkpoint in os.listdir(args.model_dir) if 'best' in checkpoint])
    best_checkpoint_path = os.path.join(args.model_dir, potential_checkpoints[-1])

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

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
    print('From Checkpoint: {}'.format(best_checkpoint_path))
