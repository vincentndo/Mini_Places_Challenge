#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time
import models
import dataset
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
parser.add_argument('--epochs', type=int, default=130,
    help='Total number of epochs to train the network')
parser.add_argument('--batch', type=int, default=256,
    help='The batch size to use for training')
parser.add_argument('--lr', type=float, default=0.01,
    help='The initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
    help='Factor by which to drop the learning rate')
parser.add_argument('--step_size', type=int, default=26,
    help='Drop the learning rate every N epochs -- this specifies N')
parser.add_argument('--momentum', type=float, default=0.9,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
    help='The L2 weight decay coefficient')
parser.add_argument('--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument('--deterministic_cudnn', action='store_true',
    help='Do not use CuDNN -- usually faster, but non-deterministic')
parser.add_argument('--gpus', type=int, nargs='+', default=[0],
    help='GPU IDs to use for training and inference '
         '(empty for CPU, i.e., just --gpus with no numbers following)')
args = parser.parse_args()

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
    print("Evaluating split \"{split}\":\t"
          "Prec@1: {top1:.2f}%\t"
          "Prec@5: {top5:.2f}%".format(
        top1=correct1 * 100 / float(total), top5=correct5 * 100 / float(total),
        split=split))

def save_model(model, epoch):
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    save_path = os.path.join(args.snapshot_dir,
                             "{}_epoch_{}.pth".format(args.snapshot_prefix, args.epoch))
    torch.save(model.state_dict(), save_path)

def load_model(model, epoch):
    load_path = os.path.join(args.snapshot_dir,
                             "{}_epoch_{}.pth".format(args.snapshot_prefix, epoch))
    model.load_state_dict(torch.load(load_path))
    return model

def train_net(model, with_val_net=True):
    # data loaders
    if with_val_net:
        val_dataset = dataset.MiniplacesDataset('val', args.crop)
        val_loader = data.DataLoader(val_dataset, batch_size=100, num_workers=4)
    train_dataset = dataset.MiniplacesDataset('train', args.crop)
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
        if with_val_net and epoch % 5 == 0:  # test on val every 5 epochs
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
                print("Epoch {epoch}/{total_epoch}:\t"
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

def test_net(model, split, k=5):
    dataset = dataset.MiniplacesDataset(split, args.crop)
    dataset.eval()
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=2)  # sequential
    model.eval()  # set model in eval mode (affect layers like dropout)
    save_file = 'top_{}_predictions.{}.csv'.format(k, split)
    with open(save_file, 'w') as f:
        f.write(','.join(['image'] + ['label%d' % i for i in range(1, k+1)]))
        f.write('\n')
        for image, input, target in dataloader:
            if args.gpus:
                input = input.cuda(args.gpus[0])
            input = Variable(input, volatile=True)
            output = model(input)
            topk = output.data.cpu().topk(k, 1, sorted=True)[1][0].numpy() # sorted top k
            f.write(''.join('%s,%s\n' % (image, ','.join(str(p) for p in preds))
                            for image, preds in zip(filenames, top5)))
    print('Predictions for split {split} dumped to: %s') % (split, save_file)

if __name__ == '__main__':
    model = models.MiniAlexNet()
    if args.gpus:
        model = model.cuda(args.gpus[0])
        if len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
    print('Training net...\n')
    train_net(model)

    print('\nTraining complete. Evaluating...\n')
    for split in ('train', 'val', 'test'):
        test_net(split)
        print()
    print('Evaluation complete.')
