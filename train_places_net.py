#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time
import models
import datasets
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

parser = argparse.ArgumentParser(
    description='Train and evaluate a net on the MIT mini-places dataset.')
parser.add_argument('--image_root', default='./images/',
    help='Directory where images are stored')
parser.add_argument('--crop', type=int, default=96,
    help=('The edge length of the random image crops'
          '(defaults to 96 for 96x96 crops)'))
parser.add_argument('--disp', type=int, default=50,
    help='Print loss/accuracy every --disp training iterations')
parser.add_argument('--snapshot_dir', default='./snapshots',
    help='Path to directory where snapshots are saved')
parser.add_argument('--snapshot_prefix', default='place_net',
    help='Snapshot filename prefix')
parser.add_argument('--epochs', type=int, default=150,
    help='Total number of epochs to train the network')
parser.add_argument('--batch', type=int, default=256,
    help='The batch size to use for training')
parser.add_argument('--lr', type=float, default=0.03,
    help='The initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
    help='Factor by which to drop the learning rate')
parser.add_argument('--step_size', type=int, default=25,
    help='Drop the learning rate every N epochs -- this specifies N')
parser.add_argument('--momentum', type=float, default=0.9,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument('--weight_decay', type=float, default=3e-4,
    help='The L2 weight decay coefficient')
parser.add_argument('--val_epoch', type=int, default=0,
    help='If positive, evaluating on val every N epochs -- this specifies N')
parser.add_argument('--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument('--deterministic_cudnn', action='store_true',
    help='Set cudnn deterministic flag. The cudnn algorithms will return '
         'deterministic results. Note that the DataLoader workers return '
         'order is still non-deterministic. To obtain deterministic results, '
         'set num_workers=0 in the DataLoader of train dataset.')
parser.add_argument('--gpus', type=int, nargs='+', default=[0],
    help='GPU IDs to use for training and inference (-1 for CPU)')
args = parser.parse_args()

# validate --gpus arg
if len(args.gpus) == 0:
    raise ValueError('GPU IDs must be specified in --gpus. '
                     'Use --gpus -1 for CPU mode')
else:
    args.gpus = list(set(args.gpus))
    devices = [-1] + list(range(torch.cuda.device_count()))
    cpu = False
    for device in args.gpus:
        cpu |= device == -1
        if device not in devices:
            raise ValueError('Unexpected device ID in --gpus. '
                             'Should be one of {}'.format(', '.join(devices)))
    if cpu and len(args.gpus) > 1:
        raise ValueError('Cannot use both GPU and CPU simultaneously in --gpus')

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.deterministic_cudnn

def accumulate_accuracy(output, target, previous=None, k=5):
    if previous is None:
        total = correct1 = correctk = 0
    else:
        total, correct1, correctk = previous
    total += output.size(0)
    topk = output.data.cpu().topk(k, 1, sorted=True)[1].numpy()  # sorted top k
    if isinstance(target, Variable):
        target = target.data
    target = target.cpu().numpy()
    correct1 += (topk[:, 0] == target).sum()
    correctk += (topk == target[:, None]).sum()
    return total, correct1, correctk

def eval_net(model, dataloader, split):
    dataloader.dataset.eval()  # use eval transform (center crop, no random flip)
    model.eval()  # set model in eval mode (affect layers like dropout)
    stats = None
    for _, input, target in dataloader:
        if args.gpus:
            input = input.cuda(args.gpus[0])
        input = Variable(input, volatile=True)
        output = model(input)
        stats = accumulate_accuracy(output, target, stats)
    total, correct1, correct5 = stats
    print("Evaluating split \"{split}\"...\t"
          "Prec@1: {top1:.2f}%\t"
          "Prec@5: {top5:.2f}%".format(
        top1=correct1 * 100 / float(total), top5=correct5 * 100 / float(total),
        split=split))

def save_model(model, epoch):
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if isinstance(model, nn.DataParallel):
        model = model.module
    save_path = os.path.join(args.snapshot_dir,
                             "{}_epoch_{}.pth".format(args.snapshot_prefix, epoch))
    torch.save(model.state_dict(), save_path)

def load_model(model, epoch):
    load_path = os.path.join(args.snapshot_dir,
                             "{}_epoch_{}.pth".format(args.snapshot_prefix, epoch))
    if isinstance(model, nn.DataParallel):
        model_ = model.module
    else:
        model_ = model
    model_.load_state_dict(torch.load(load_path))
    return model

def train_net(model):
    # data loaders
    if args.val_epoch > 0:
        val_dataset = datasets.MiniplacesDataset('val', args.crop)
        val_loader = data.DataLoader(val_dataset, batch_size=100, num_workers=4)
    train_dataset = datasets.MiniplacesDataset('train', args.crop)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    # optimizer and lr scheduler
    # seaprate weights and bias so we can double lr and have 0 decay for bias
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if name.endswith('weight'):
            weights.append(param)
        elif name.endswith('bias'):
            biases.append(param)
        else:
            raise ValueError('unexpected parameter ' + name)
    optimizer = optim.SGD([
            dict(params=weights, lr=args.lr, weight_decay=args.weight_decay),
            dict(params=biases, lr=args.lr * 2, weight_decay=0),
        ], momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
    # loss
    criterion = nn.CrossEntropyLoss()
    # train
    for epoch in range(args.epochs):
        if args.val_epoch > 0 and epoch % args.val_epoch == 0:
            eval_net(model, val_loader, 'val')
        save_model(model, epoch)
        scheduler.step()
        model.train()  # set model in training mode (affects layers like dropout)
        stats = None
        time_start = time.time()
        for iteration, (_, input, target) in enumerate(train_loader):
            time_start = time.time()
            # prepare data (pull to gpu, wrap in Variable)
            if args.gpus:
                input = input.cuda(args.gpus[0])
                target = target.cuda(args.gpus[0])
            input = Variable(input)
            target = Variable(target)
            # train step
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()   # compute gradient
            optimizer.step()  # optimizer step
            stats = accumulate_accuracy(output, target)
            time_end = time.time()
            if (args.disp > 0) and (iteration % args.disp == 0):
                total, correct1, correct5 = stats
                stats = None
                print("Epoch {epoch}/{total_epoch}\t"
                      "Iteration {it}/{total_it} ({time:.2f} s/it)\t"
                      "Loss: {loss:.4f}\t"
                      "Prec@1: {top1:.2f}%\t"
                      "Prec@5: {top5:.2f}%".format(
                      epoch=epoch, total_epoch=args.epochs,
                      it=iteration, total_it=len(train_loader),
                      time=time_end - time_start, loss=loss.data[0],
                      top1=correct1 * 100 / float(total),
                      top5=correct5 * 100 / float(total)))
            time_start = time_end

    eval_net(model, train_loader, 'train')
    if with_val_net:
        eval_net(model, val_loader, 'val')
    save_model(model, args.epochs)

def test_net(model, k=5):
    dataset = datasets.MiniplacesDataset('test', args.crop)
    dataset.eval()
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=2)  # sequential
    model.eval()  # set model in eval mode (affect layers like dropout)
    save_file = 'top_{}_predictions.test.csv'.format(k)
    with open(save_file, 'w') as f:
        f.write(','.join(['image'] + ['label%d' % i for i in range(1, k+1)]))
        f.write('\n')
        for image, input in dataloader:
            if args.gpus:
                input = input.cuda(args.gpus[0])
            input = Variable(input, volatile=True)
            output = model(input)
            topk = output.data.cpu().topk(k, 1, sorted=True)[1][0].numpy() # sorted top k
            f.write('{},{}\n'.format(image[0], ','.join(str(p) for p in topk)))
    print('Predictions for split "test" dumped to: {}'.format(save_file))

if __name__ == '__main__':
    model = models.MiniAlexNet()
    if args.gpus:
        model = model.cuda(args.gpus[0])
        if len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
    print('Training net...\n')
    train_net(model)
    print('\nTraining complete. Evaluating...\n')
    test_net(model)
    print('Evaluation complete.')

