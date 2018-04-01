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
parser.add_argument('--epochs', type=int, default=128,
    help='Total number of epochs to train the network')
parser.add_argument('--batch', type=int, default=256,
    help='The batch size to use for training')
parser.add_argument('--lr', type=float, default=0.1,
    help='The initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
    help='Factor by which to drop the learning rate')
parser.add_argument('--step_size', type=int, default=5,
    help='Drop the learning rate every N epochs -- this specifies N')
parser.add_argument('--momentum', type=float, default=0.9,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
    help='The L2 weight decay coefficient')
parser.add_argument('--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument('--no_cudnn', action='store_true',
    help='Do not use CuDNN -- usually faster, but non-deterministic')
parser.add_argument('--gpu', type=int, default=0,
    help='GPU ID to use for training and inference (-1 for CPU)')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = not args.no_cudnn


def val_net(model, epoch, total_epoch, val_loader):
    model.eval()
    total = correct1 = correct5 = 0
    for input, target in val_loader:
        if args.gpu >= 0:
            input = input.cuda(args.gpu)
        input = Variable(input, volatile=True)
        target = target.numpy()
        output = model(input)
        top5 = output.topk(5, 1)[1].data.cpu().numpy()
        correct1 += (top5[:, 0] == target).sum()
        correct5 += (top5 == target[:, None]).sum()
        total += input.size(0)
    print("Epoch {epoch}/{total_epoch}:\t"
          "Top 1 accuracy: {top1_p:.2f}% ({top1}/{total})\t"
          "Top 5 accuracy: {top5_p:.2f}% ({top5}/{total})".format(
        top1_p=correct1 * 100 / float(total), top1=correct1,
        top5_p=correct5 * 100 / float(total), top5=correct5,
        total=total, epoch=epoch, total_epoch=total_epoch))

def train_net(model, with_val_net=True):
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    # data loaders
    if with_val_net:
        val_dataset = dataset.MiniplacesDataset('val', args.crop)
        val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=4)
    train_dataset = dataset.MiniplacesDataset('train', args.crop)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    # optimizer and lr scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
    # loss
    criterion = nn.CrossEntropyLoss()
    # train
    for epoch in range(args.epochs):
        if with_val_net and epoch % 1 == 0:  # test on val every 5 epochs
            val_net(model, epoch, args.epochs, val_loader)
        save_path = os.path.join(args.snapshot_dir,
                                 "{}_epoch_{}.pth".format(args.snapshot_prefix, epoch))
        torch.save(model.state_dict(), save_path)
        scheduler.step()
        model.train()
        for iteration, (input, target) in enumerate(train_loader):
            time_start = time.time()
            # prepare data (pull to gpu, wrap in Variable)
            if args.gpu >= 0:
                input = input.cuda(args.gpu)
                target = target.cuda(args.gpu)
            input = Variable(input)
            target = Variable(target)
            # train step
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()   # compute gradient
            optimizer.step()  # optimizer step
            time_used = time.time() - time_start
            if (args.disp > 0) and (iteration % args.disp == 0):
                print("Epoch {epoch}/{total_epoch}:\t"
                      "Iteration {it}/{total_it} ({time:.2f} s/it)\t"
                      "Loss: {loss:.4f}\t".format(
                      epoch=epoch, total_epoch=args.epochs,
                      it=iteration, total_it=len(train_loader),
                      time=time_used, loss = loss.data[0]))

    if with_val_net:
        val_net(model, args.epochs, args.epochs, val_loader)
    save_path = os.path.join(args.snapshot_dir,
                             "{}_epoch_{}_final.pth".format(args.snapshot_prefix, args.epochs))
    torch.save(model.state_dict(), save_path)

# def eval_net(split, K=5):
#     print 'Running evaluation for split:', split
#     filenames = []
#     labels = []
#     split_file = get_split(split)
#     with open(split_file, 'r') as f:
#         for line in f.readlines():
#             parts = line.split()
#             assert 1 <= len(parts) <= 2, 'malformed line'
#             filenames.append(parts[0])
#             if len(parts) > 1:
#                 labels.append(int(parts[1]))
#     known_labels = (len(labels) > 0)
#     if known_labels:
#         assert len(labels) == len(filenames)
#     else:
#         # create file with 'dummy' labels (all 0s)
#         split_file = to_tempfile(''.join('%s 0\n' % name for name in filenames))
#     test_net_file = miniplaces_net(split_file, train=False, with_labels=False)
#     weights_file = snapshot_at_iteration(args.iters)
#     net = caffe.Net(test_net_file, weights_file, caffe.TEST)
#     top_k_predictions = np.zeros((len(filenames), K), dtype=np.int32)
#     if known_labels:
#         correct_label_probs = np.zeros(len(filenames))
#     offset = 0
#     while offset < len(filenames):
#         probs = net.forward()['probs']
#         for prob in probs:
#             top_k_predictions[offset] = (-prob).argsort()[:K]
#             if known_labels:
#                 correct_label_probs[offset] = prob[labels[offset]]
#             offset += 1
#             if offset >= len(filenames):
#                 break
#     if known_labels:
#         def accuracy_at_k(preds, labels, k):
#             assert len(preds) == len(labels)
#             num_correct = sum(l in p[:k] for p, l in zip(preds, labels))
#             return num_correct / len(preds)
#         for k in [1, K]:
#             accuracy = 100 * accuracy_at_k(top_k_predictions, labels, k)
#             print '\tAccuracy at %d = %4.2f%%' % (k, accuracy)
#         cross_ent_error = -np.log(correct_label_probs).mean()
#         print '\tSoftmax cross-entropy error = %.4f' % (cross_ent_error, )
#     else:
#         print 'Not computing accuracy; ground truth unknown for split:', split
#     filename = 'top_%d_predictions.%s.csv' % (K, split)
#     with open(filename, 'w') as f:
#         f.write(','.join(['image'] + ['label%d' % i for i in range(1, K+1)]))
#         f.write('\n')
#         f.write(''.join('%s,%s\n' % (image, ','.join(str(p) for p in preds))
#                         for image, preds in zip(filenames, top_k_predictions)))
#     print 'Predictions for split %s dumped to: %s' % (split, filename)

if __name__ == '__main__':
    model = models.MiniAlexNet()
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
    print('Training net...\n')
    train_net(model)

    # print '\nTraining complete. Evaluating...\n'
    # for split in ('train', 'val', 'test'):
    #     eval_net(split)
    #     print
    # print 'Evaluation complete.'
