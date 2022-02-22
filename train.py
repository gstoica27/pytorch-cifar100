""" train network using pytorch

author baiyu
"""


import pdb
from ast import parse, literal_eval
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.csam import ConvAttnWrapper
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, read_yaml, \
    save_yaml, name_model

def train(epoch):

    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if para.grad is not None:
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if batch_index % 10 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        try:
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        except:
            import pdb; pdb.set_trace()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct_1 = 0.0
    correct_5 = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        # import pdb; pdb.set_trace()
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1
        correct_1 += correct[:, :1].sum()

        test_loss += loss.item()
        # _, preds = outputs.max(1)
        # correct += preds.eq(labels).sum()

    top1_accuracy = correct_1 / len(cifar100_test_loader.dataset)
    top5_accuracy = correct_5 / len(cifar100_test_loader.dataset)
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top-1 Accuracy: {:.4f}, Top-5 Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        top1_accuracy,
        top5_accuracy,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Top1_Accuracy', top1_accuracy, epoch)
        writer.add_scalar('Test/Top5_Accuracy', top5_accuracy, epoch)

    return top1_accuracy, top5_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    # Redundant with convattn.yaml
    parser.add_argument('--approach_name', help='name of the approach: "1", "2", "3", "4". Look at csam.py for full documentation about each of the methods.')
    parser.add_argument('--suffix', help='Suffix used to create the directory where the model weights/logs will be stored')
    parser.add_argument('--pos_emb_dim', help='Dimension of position embedding')
    parser.add_argument('--softmax_temp', help='Softmax Temperature')
    parser.add_argument('--injection_info', help='[[InjectionLayer, NumStack, FilterSize], [...]]')
    parser.add_argument('--stride', help='Size of the stride')
    parser.add_argument('--apply_stochastic_stride', action='store_true', default=None, help='Apply stochastic stride')
    parser.add_argument('--use_residual_connection', action='store_true', default=None, help='Use residual connection')

    parser.add_argument(
        '-variant_config_path',
        type=str,
        default='configs/convattn.yaml',
        help='path to variant configuration'
    )
    parser.add_argument('-naming_suffix', type=str, default='', help='Add suffix to model name')
    args = parser.parse_args()
    variant_config = read_yaml(args.variant_config_path)
    print(variant_config)
    net = get_network(args)

    variant_config = read_yaml(args.variant_config_path)

    # Overwrite all file configs with the ones set inline
    if args.approach_name is not None:
        variant_config["approach_name"] = args.approach_name
    if args.suffix is not None:
        variant_config["suffix"] = args.suffix
    if args.pos_emb_dim is not None:
        variant_config["pos_emb_dim"] = int(args.pos_emb_dim)
    if args.softmax_temp is not None:
        variant_config["softmax_temp"] = float(args.softmax_temp)
    if args.injection_info is not None:
        variant_config["injection_info"] = literal_eval(args.injection_info)
    if args.stride is not None:
        variant_config["stride"] = int(args.stride)
    if args.apply_stochastic_stride is not None:
        variant_config["apply_stochastic_stride"] = args.apply_stochastic_stride
    if args.use_residual_connection is not None:
        variant_config["use_residual_connection"] = args.use_residual_connection

    model = ConvAttnWrapper(backbone=net, variant_kwargs=variant_config).to('cuda:0')

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    model_name = name_model(variant_config)
    if 'suffix' in variant_config and variant_config['suffix'] not in {None, ''}:
        model_name += '_{}'.format(variant_config['suffix'])
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net, variant_config["approach_name"], model_name), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_dir = os.path.join(settings.CHECKPOINT_PATH, args.net, variant_config["approach_name"], model_name, recent_folder)

    else:
        checkpoint_dir = os.path.join(settings.CHECKPOINT_PATH, args.net, variant_config["approach_name"], model_name, settings.TIME_NOW)

    with open('logs/started.txt', 'a') as f:
        f.write(checkpoint_dir)
        f.write("\n")
    print('Saving to: {}'.format(checkpoint_dir))

    try:

        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)

        #since tensorboard can't overwrite old values
        #so the only way is to create a new tensorboard log
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, model_name, settings.TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32)
        if args.gpu:
            input_tensor = input_tensor.cuda()
        writer.add_graph(model, input_tensor)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(checkpoint_dir)
        save_yaml(
            path=os.path.join(checkpoint_dir, 'convattn.yaml'),
            data=variant_config
        )
        checkpoint_path = os.path.join(checkpoint_dir, '{net}-{epoch}-{type}.pth')

        best_top1 = 0.0
        best_top5_at_top1 = 0.0
        best_epoch = 0
        if args.resume:
            best_weights = best_acc_weights(checkpoint_dir)
            if best_weights:
                weights_path = os.path.join(checkpoint_dir, best_weights)
                print('found best acc weights file:{}'.format(weights_path))
                print('load best training file to test acc...')
                model.load_state_dict(torch.load(weights_path))
                best_top1, best_top5 = eval_training(tb=False)
                print('best top1 is {:0.2f} | best top5 is: {:0.2f}'.format(best_top1, best_top5))

            recent_weights_file = most_recent_weights(checkpoint_dir)
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(checkpoint_dir, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            model.load_state_dict(torch.load(weights_path))
            resume_epoch = last_epoch(checkpoint_dir)

        for epoch in range(1, settings.EPOCH + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            train(epoch)
            test_top1, test_top5 = eval_training(epoch)

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_top1 < test_top1:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                with open('logs/latest_successful_checkpoint_paths.txt', 'a') as f:
                    f.write(weights_path)
                    f.write("\n")
                print('saving weights file to {}'.format(weights_path))
                torch.save(model.state_dict(), weights_path)
                best_top1 = test_top1
                best_top5_at_top1 = test_top5
                best_epoch = epoch
                continue
            print()
            print('Best Point | Epoch: {} | Top-1: {:.4f} | Top-5: {:.4f}'.format(best_epoch, best_top1, best_top5_at_top1))
            print('Best Error | Epoch: {} | Top-1: {:.4f} | Top-5: {:.4f}'.format(best_epoch, 1-best_top1, 1-best_top5_at_top1))
            print()
            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                with open('logs/latest_successful_checkpoint_paths.txt', 'a') as f:
                    f.write(weights_path)
                    f.write("\n")
                torch.save(model.state_dict(), weights_path)

        writer.close()

    except Exception as e:
        print(e)
        with open('logs/latest_failed_checkpoint_paths.txt', 'a') as f:
            f.write(checkpoint_path)
            f.write("\n")
