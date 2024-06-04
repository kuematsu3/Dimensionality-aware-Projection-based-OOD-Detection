import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys, os, glob, argparse, math
import numpy as np
# from utils import create_model
from utils import *





def train(net, loader, device, criterion, optimizer, scheduler, nc):

    loss_sum, acc_sum, len_sum = 0, 0, 0
    log0 = []
    net.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        # initialization
        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        len0 = len(inputs)

        logits = net(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        _, predicted = logits.max(1)
        acc = predicted.eq(targets).sum().item()

        log0.append([lr, len0, loss.item(), acc])
        loss_sum += (loss.item() * len0)
        acc_sum += acc
        len_sum += len0

    inv = 1.0 / len_sum
    loss_sum *= inv
    acc_sum *= inv
    print(f'{loss_sum:8.5f}, {acc_sum:8.5f} |', end='')

    return loss_sum, acc_sum, np.array(log0, dtype=np.float32)



def test(net, loader, device, criterion, nc, save_logits=False, save_targets=False):

    net.eval()
    test_loss, test_acc, total = 0, 0, 0
    logits_all, targets_all = [], []
    for batch_idx, (inputs, targets) in enumerate(loader):
        # communicating data to gpu
        inputs, targets = inputs.to(device), targets.to(device)
        len0 = len(inputs)

        with torch.no_grad():
            logits = net(inputs)

        loss = criterion(logits, targets)
        _, predicted = logits.max(1)
        acc = predicted.eq(targets).sum().item()

        test_loss += (loss.item() * len0)
        test_acc += acc
        total += len0

        if save_logits:  logits_all.append(logits)
        if save_targets:  targets_all.append(targets)

    inv = 1.0 / total
    test_loss *= inv
    test_acc *= inv
    print(f'{test_loss:8.5f}, {test_acc:8.5f} |', end='')
    if save_logits:  logits_all = torch.cat(logits_all, 0).detach().to('cpu').clone().numpy()
    if save_targets:  targets_all = torch.cat(targets_all, 0).detach().to('cpu').clone().numpy()

    return test_loss, test_acc, logits_all, targets_all


def prepare_data(dataset, batchsize, train_ratio=1.0):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform0 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(),
    ])
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    transform0_gray = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(),
    ])
    transform1_gray = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean, std),
    ])

    dir0 = "/home/user/"
    if dataset == 'cifar10':
        trainset0 = torchvision.datasets.CIFAR10(
            root=dir0+'data', train=True, download=True, transform=transform0)
        testset = torchvision.datasets.CIFAR10(
            root=dir0+'data', train=False, download=True, transform=transform1)
    elif dataset == 'svhn':
        trainset0 = torchvision.datasets.SVHN(
            root=dir0+'data', split='train', download=True, transform=transform0)
        testset = torchvision.datasets.SVHN(
            root=dir0+'data', split='test', download=True, transform=transform1)
    elif dataset == 'mnist':
        trainset0 = torchvision.datasets.MNIST(
            root=dir0+'data', train=True, download=True, transform=transform0_gray)
        testset = torchvision.datasets.MNIST(
            root=dir0+'data', train=False, download=True, transform=transform1_gray)
    elif dataset == 'cifar100':
        trainset0 = torchvision.datasets.CIFAR100(
            root=dir0+'data', train=True, download=True, transform=transform0)
        testset = torchvision.datasets.CIFAR100(
            root=dir0+'data', train=False, download=True, transform=transform1)
    else:
        assert False, 'dataset = {} is invalid'.format(dataset)


    num_trainset0 = len(trainset0)
    num_trainset = int(num_trainset0*train_ratio)
    num_validset = num_trainset0 - num_trainset
    num_testset = len(testset)

    trainset, validset = torch.utils.data.random_split(trainset0, [num_trainset, num_validset])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batchsize, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False, num_workers=2)

    print('number of data for each dataset: ', f'{num_trainset} train, ', 
          f'{num_validset} valid, ', f'{num_testset} test')
    num_iteration = len(trainset) // batchsize + 1
    print('number of iterations per epoch = ', num_iteration)

    return trainloader, validloader, testloader, num_iteration


def parse_args(argv):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model-name', default='resnet18', help='model name')

    parser.add_argument('--traindata', default='cifar10', help='train dataset')
    parser.add_argument('--train-batchsize', default=128, type=int, help='training batch-size')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=1200, type=int, help='training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch for training')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in total')

    return parser.parse_args(argv)


def file_output(net, optimizer, scheduler, epoch):

    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint'):  os.mkdir('checkpoint')

    emod = epoch % 2
    torch.save(state, f'./checkpoint/ckpt-log{emod}.pth')
    with open('./epoch_tmp.d', mode='w') as f:
        f.write(str(epoch))

    if epoch % 200 == 0:
        print('save the network and others')
        torch.save(state, f'./checkpoint/ckpt-{epoch:0>4}.pth')
#


def print_parametercheck(args):
    print('=== {} ==='.format(args.model_name))
    print('weight decay: {:f}'.format(args.wdecay))
    print('total epochs: {:d}'.format(args.epochs))
    print('start epochs: {:d}'.format(args.start_epoch))
    print('lr: {:f}'.format(args.lr))
    print('batch size: {:d}'.format(args.train_batchsize))
    print('traindata: {:s}'.format(args.traindata))

    return
#

def set_model(args, device, num_iteration, num_classes):
    net = create_model(args.model_name, num_classes=num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*num_iteration)

    epoch0 = args.start_epoch  # start from a specified epoch or the last checkpoint epoch
    if args.resume:
        # Load checkpoint.
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        if epoch0 != 0:
            checkpoint = torch.load(f'./checkpoint/ckpt-{epoch0:0>4}.pth')
        else:
            with open('./epoch_tmp.d') as f:
                epoch0 = f.read()
                epoch0 = int(epoch0)
            emod = epoch0 % 2
            checkpoint = torch.load(f'./checkpoint/ckpt-log{emod}.pth')

        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    epoch0 = epoch0 + 1

    return net, optimizer, scheduler, epoch0
#


def main():
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_lr = args.lr * args.train_batchsize / 64.0
    print('device: ', device)

    trainloader, validloader, testloader, num_iteration = prepare_data(
        args.traindata, args.train_batchsize)
    net, optimizer, scheduler, epoch0 = set_model(args, device, num_iteration, args.num_classes)
    print('model prepared')
    print(net)

    criterion = nn.CrossEntropyLoss()

    print('     |    Training   |      Test     ')
    print('Epoch| loss, accuracy| loss, accuracy')

    for epoch in range(epoch0, args.epochs+1):
        print('{:>5}|'.format(epoch), end='')
        loss0, acc0, log0 = train(net, trainloader, device, criterion, optimizer, scheduler, args.num_classes)
        loss1, acc1, _, _ = test(net, testloader, device, criterion, args.num_classes)
        print()
        if not os.path.isdir('train_log'):  os.mkdir('train_log')
        np.savez_compressed(f'./train_log/train_epoch{epoch:0>4}', log0)
        file_output(net, optimizer, scheduler, epoch)

    ### Save checkpoint
    file_output(net, optimizer, scheduler, epoch)


if __name__ == '__main__':
    main()
