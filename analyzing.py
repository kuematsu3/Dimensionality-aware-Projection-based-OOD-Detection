import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import sys, os, glob, argparse, math, time
import numpy as np
from utils import *

from features import *
from projections import *
from kernels import *



def evaluate_propagations(net, dataloader, data0, device, layers, layers_bn, layers0, params, kernel1, startime,
                          args, datasets, dataset, epoch0, imodel,
                          save_targets=False, compute_features=False, compute_gram=False, compute_convproject=False):

    if compute_convproject:
        conv_projection(args.model_name, net, kernel1, dataloader, device, params, layers, epoch0, dataset, 'all', imodel)
        print('projected to compressed net', time.perf_counter()-startime)

    _, _ = test(args.model_name, net, dataloader, device, layers, params, dataset, epoch0, imodel, args.num_classes,
                   save_logits=True, save_targets=save_targets, compute_features=compute_features, save_norm00=True, save_norm0=True)
    print('test for original net performed', time.perf_counter()-startime)

    if compute_gram:
        compute_gram_matrix(args.model_name, net, layers, data0, device, dataset, epoch0, imodel)
        print('gram matrix for original net computed', time.perf_counter()-startime)

    _, _ = test(args.model_name, net, dataloader, device, layers_bn, params, dataset,
                   epoch0, imodel+'-bn', args.num_classes, save_norm00=True)
    print('norm before bn computed', time.perf_counter()-startime)

    if compute_convproject:
        conv_projection(args.model_name, net, kernel1, dataloader, device, params, layers, epoch0, dataset, imodel, imodel)
        print('projected to compressed net', time.perf_counter()-startime)

    check_noisestability(args.model_name, net, datasets, dataset, data0, args, layers0, device, params, epoch0, imodel)
    print('noise stability computed', time.perf_counter()-startime)


    return



def check_noisestability(model, net, datasets, dataset0, data0, args, layers0, device, params, epoch0, imodel):
    dataset1 = dataset0
    if dataset1=='traindata':
        dataloader1, _, _ = prepare_data_eval(args.traindata, args.batchsize)
    else:
        _, dataloader1, _ = prepare_data_eval(dataset1, args.batchsize)

    data1, label1 = prepare_data_topn(dataloader1, args.batchsize, args.num_stability)
    print(dataset0, dataset1, len(data0), len(data1))

    test_noiseprop(model, net, data0, data1, device, layers0, params,
                   dataset0, dataset1, epoch0, imodel, eta=0.1)

    return


def test_noiseprop(model, net, data0, data1, device, layers0, params, dataset0, dataset1, epoch0, imodel, eta=0.1):

    stability = []
    eta = torch.from_numpy(np.array(eta, dtype=np.float32)).clone().to(device)
    net.eval()
    for batch_idx, (inputs0, inputs1) in enumerate(zip(data0, data1)):
        inputs0 = inputs0.to(device)
        inputs1 = inputs1.to(device)
        len0, len1 = len(inputs0), len(inputs1)

        with torch.no_grad():  logits0, features0 = partialprop(model, net, inputs0, 0, layers0)
        features1 = [torch.randn(feature.shape).to(device) for feature in features0]

        norm0 = []
        for idx0 in range(len(features0)):
            if  params[layers0[idx0]][5]:
                tmp0 = torch.sum(features0[idx0]*features0[idx0], (1,2,3), keepdim=True)
                tmp1 = torch.sum(features1[idx0]*features1[idx0], (1,2,3), keepdim=True)
            else:
                tmp0 = torch.sum(features0[idx0]*features0[idx0], 1, keepdim=True)
                tmp1 = torch.sum(features1[idx0]*features1[idx0], 1, keepdim=True)
            
            features1[idx0] = features1[idx0] * torch.sqrt(tmp0/tmp1)
            norm0.append(tmp0.reshape(len0))

        tmp0 = torch.sum(logits0*logits0, 1)
        norm0.append(tmp0)

        perturbed = []
        for idx0, (layer, feature0, feature1) in enumerate(zip(layers0, features0, features1)):
            feature = feature0 + feature1 * eta
            with torch.no_grad():  logits_perturbed, features_perturbed = partialprop(model, net, feature, layer, layers0)
            diff = logits_perturbed - logits0
            diff = torch.sum(diff*diff, 1) / norm0[-1]
            noised0 = [diff] # 0: output, 1: layers0[-1], ..., len(layers0)-idx0: layers0[idx0]
            for idx1 in range(len(features_perturbed)):
                diff = features_perturbed[-(1+idx1)] - features0[-(1+idx1)]
                diff = torch.sum((diff*diff).reshape(len0, -1), 1) / norm0[-(2+idx1)]
                noised0.append(diff)
            perturbed.append(noised0)

        stability.append(perturbed)


    name0 = f'./projections/stability{epoch0-1:0>4}_'
    name1 = '_'+imodel+'_'+dataset0+'-'+dataset1
    np.set_printoptions(formatter={'float': '{:.2e}'.format})
    for idx0 in range(len(layers0)):
        diff_layers = []
        for idx1 in range(len(layers0)-idx0+1):
            diff = torch.cat([feature[idx0][len(layers0)-idx0-idx1] for feature in stability], 0)
            diff_layers.append(diff.detach().to('cpu').clone().numpy())

        diff_layers = np.array(diff_layers, dtype=np.float32)
        np.savez_compressed(name0+f'{layers0[idx0]:0>2}-eta{eta:.3f}'+name1, diff_layers)

    np.set_printoptions(formatter=None)

    return 



def test(model, net, loader, device, layers, params, dataset, epoch0, imodel, nc, 
         save_logits=False, save_targets=False, compute_features=False,
         save_features=False, save_eigvecs=False, save_norm00=False, save_norm0=False):

    layers0 = [int(str(layer).replace('_bn','')) for layer in layers]

    net.eval()
    logits_all, targets_all, features_all, norms00, norms0 = [], [], [], [], []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        len0 = len(inputs)

        with torch.no_grad():  logits, features = partialprop(model, net, inputs, 0, layers)

        if save_logits:  logits_all.append(logits.detach().to('cpu').clone().numpy())
        targets_all.append(targets.detach().to('cpu').clone().numpy())

        if compute_features:
            avg = [avg_feature(feature, params[layer], device) for layer, feature in zip(layers0, features)]
            features_all.append([[avg0[0].detach().to('cpu').clone().numpy()] for avg0 in avg])
            # print(batch_idx, len(avg), [len(avg0) for avg0 in avg], [avg0[0].shape for avg0 in avg])
        if save_norm00 or save_norm0:
            norm00, norm0 = get_norm0(features, layers0, params)
            if save_norm00:  norms00.append([norm.detach().to('cpu').clone().numpy() for norm in norm00])
            if save_norm0:  norms0.append([norm.detach().to('cpu').clone().numpy() for norm in norm0])


    if not os.path.isdir('outputs'):  os.mkdir('outputs')
    if not os.path.isdir('projections'):  os.mkdir('projections')
    if not os.path.isdir('features'):  os.mkdir('features')

    if save_logits:
        logits_all = np.concatenate(logits_all, 0)
        np.savez_compressed(f'./outputs/logits{epoch0-1:0>4}_{imodel}_'+dataset, logits_all)

    targets_all = np.concatenate(targets_all, 0)
    if save_targets:  np.savez_compressed(f'./outputs/labels{epoch0-1:0>4}_'+dataset, targets_all)

    if save_norm00:
        norms = [np.concatenate([norm[idx] for norm in norms00], 0) for idx in range(len(layers0))]
        for layer, norm in zip(layers0, norms):
            np.savez_compressed(f'./projections/norm00-{epoch0-1:0>4}-{layer:0>2}_{imodel}_'+dataset, norm)

    if save_norm0:
        norms = [np.concatenate([norm[idx] for norm in norms0], 0) for idx in range(len(layers0))]
        for layer, norm in zip(layers0, norms):
            np.savez_compressed(f'./projections/norm0-{epoch0-1:0>4}-{layer:0>2}_{imodel}_'+dataset, norm)


    features = []
    if compute_features:
        if len(layers0) < 4:
            print('concatenating features')
            features = [np.concatenate([features[idx][0] for features in features_all], 0) \
                        for idx in range(len(layers0))]
        if dataset=='traindata':
            for idx, layer in enumerate(layers0):
                feature = np.concatenate([features[idx][0] for features in features_all], 0)
                covt = compute_covariances_np([layer], [feature], targets_all, nc, epoch0, imodel,
                                              'gauss', device, save_eigvecs=save_eigvecs)
                np.savez_compressed(f'./features/sigma{epoch0-1:0>4}-{layer:0>2}_{imodel}_'+dataset, tied=covt[0][1])

    return targets_all, features





def prepare_data_eval(dataset, batchsize):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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
            root=dir0+'data', train=True, download=True, transform=transform1)
        testset = torchvision.datasets.CIFAR10(
            root=dir0+'data', train=False, download=True, transform=transform1)
    elif dataset == 'svhn':
        trainset0 = torchvision.datasets.SVHN(
            root=dir0+'data', split='train', download=True, transform=transform1)
        testset = torchvision.datasets.SVHN(
            root=dir0+'data', split='test', download=True, transform=transform1)
    elif dataset == 'mnist':
        trainset0 = torchvision.datasets.MNIST(
            root=dir0+'data', train=True, download=True, transform=transform1_gray)
        testset = torchvision.datasets.MNIST(
            root=dir0+'data', train=False, download=True, transform=transform1_gray)
    elif dataset == 'cifar100':
        trainset0 = torchvision.datasets.CIFAR100(
            root=dir0+'data', train=True, download=True, transform=transform1)
        testset = torchvision.datasets.CIFAR100(
            root=dir0+'data', train=False, download=True, transform=transform1)
    else:
        assert False, 'dataset = {} is invalid'.format(dataset)

    trainloader = torch.utils.data.DataLoader(
        trainset0, batch_size=batchsize, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False, num_workers=2)

    print('number of data for each dataset: ', f'{len(trainset0)} train, ', f'{len(testset)} test')
    num_iteration = len(trainset0) // batchsize + 1
    print('number of iterations per epoch = ', num_iteration)
    return trainloader, testloader, num_iteration


def prepare_data_topn(loader, batchsize, topn):
    len_tot = 0
    inputs_all, targets_all = [], []
    for idx, (inputs, targets) in enumerate(loader):
        len0 = len(inputs)
        len_tot += len0
        if len_tot < topn:
            inputs_all.append(inputs)
            targets_all.append(targets)
        else:
            inputs_all.append(inputs[:len0+topn-len_tot])
            targets_all.append(targets[:len0+topn-len_tot])
    inputs_all = torch.cat(inputs_all, 0)
    targets_all = torch.cat(targets_all, 0)

    num_iteration = (topn + batchsize - 1) // batchsize
    inputs, targets = [], []
    for idx in range(num_iteration):
        idx0 = idx * batchsize;  idx1 = min(idx0+batchsize, topn)
        inputs.append(inputs_all[idx0:idx1]);  targets.append(targets_all[idx0:idx1])
    print(f'top+{topn}: ', len(loader), batchsize, len(inputs), len(targets))

    return inputs, targets



def parse_args(argv):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model-name', default='resnet18', help='model name')

    parser.add_argument('--traindata', default='cifar10', help='train dataset')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size for train and test')

    parser.add_argument('--lr', default=0.5, type=float, help='learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=1200, type=int, help='training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch for training')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in total')
    parser.add_argument('--num_grams', default=10000, type=int, help='number of data for gram matrices')
    parser.add_argument('--num_stability', default=10000, type=int, help='number of data for noise stability')

    return parser.parse_args(argv)


# in_channel, out_channel, kernel_size, padding, stride, is_conv
def set_param(modelname, nc):
    if modelname=='vgg13':
        params = [  [3, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True],
                    [128, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True],
                    [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True], [512, nc, 1, 1, 1, False]  ]
    elif modelname=='vgg16':
        params = [  [3, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True],
                    [128, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 512, 3, 1, 1, True],
                    [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True],
                    [512, 512, 3, 1, 1, True], [512, nc, 1, 1, 1, False]  ]
    elif modelname=='resnet18': # block3, block 5, block7
        params = [  [3, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True],
                    [64, 64, 3, 1, 1, True], [64, 128, 3, 1, 2, True], [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True],
                    [128, 128, 3, 1, 1, True], [128, 256, 3, 1, 2, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True],
                    [256, 256, 3, 1, 1, True], [256, 512, 3, 1, 2, True], [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True],
                    [512, 512, 3, 1, 1, True], [512, nc, 1, 1, 1, False]  ]
    elif modelname=='resnet34': # block4, block8, block14
        params = [  [3, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True],
                    [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 64, 3, 1, 1, True], [64, 128, 3, 1, 2, True],
                    [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True],
                    [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True], [128, 128, 3, 1, 1, True], [128, 256, 3, 1, 2, True],
                    [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True],
                    [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True],
                    [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 256, 3, 1, 1, True], [256, 512, 3, 1, 2, True],
                    [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True], [512, 512, 3, 1, 1, True],
                    [512, 512, 3, 1, 1, True], [512, nc, 1, 1, 1, False]  ]
    #
    return params


def set_layers_model(modelname):
    if modelname=='vgg13':
        layers = [i for i in range(11)]
    elif modelname=='vgg16':
        layers = [i for i in range(14)]
    elif modelname=='resnet18': # block3, block 5, block7
        layers = [0] + [i for i in range(1, 18, 2)]
    elif modelname=='resnet34': # block4, block8, block14
        layers = [0] + [i for i in range(1, 34, 2)]
    #
    return layers


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


def print_parametercheck(args):
    print('=== {} ==='.format(args.model_name))
    print('traindata: {:s}'.format(args.traindata))
    print('total epochs: {:d}'.format(args.epochs))
    print('start epochs: {:d}'.format(args.start_epoch))
    print('lr: {:f}'.format(args.lr))
    print('batch size: {:d}'.format(args.batchsize))
    print('weight decay: {:f}'.format(args.wdecay))

    return
#


def main_analyzing():
    startime = time.perf_counter()
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_lr = args.lr * args.batchsize / 64.0

    trainloader, _, num_iteration = prepare_data_eval(args.traindata, args.batchsize)
    net, optimizer, scheduler, epoch0 = set_model(args, device, num_iteration, args.num_classes)
    net0, optimizer, scheduler, epoch0 = set_model(args, device, num_iteration, args.num_classes)
    params = set_param(args.model_name, args.num_classes)

    ratios = [1e-6, 1e-5, 1e-4] + [10**(i*0.1-3) for i in range(10)] + [10**(i*0.05-2) for i in range(40)]
    datasets = ['traindata', 'cifar10', 'svhn', 'mnist', 'cifar100']

    weightname0, weightname = get_weight_dict_name(args.model_name)
    ndims, svals, kernels1, kernels0 = get_parameters(net, weightname, ratios) # ndim, proj, compress
    ndims = np.array(ndims)
    svals0 = np.zeros((len(svals), max([len(sval) for sval in svals])))
    for i, sval in enumerate(svals):  svals0[i,0:len(sval)] = sval[0:len(sval)]
    np.savez_compressed(f'./svals{epoch0-1:0>4}', svals=svals0, ndims=ndims, ratios=np.array(ratios))

    layers = [idx for idx in range(len(weightname))]
    layers_bn = [f'{idx}_bn' for idx in range(len(weightname)-1)]
    layers0 = set_layers_model(args.model_name)

    print(layers)
    print(layers0)
    print('parameters prepared', svals0.shape, np.array(ndims).shape, len(ratios), time.perf_counter()-startime)


    for dataset in datasets:
        if dataset=='traindata':
            dataloader = trainloader
        else:
            _, dataloader, _ = prepare_data_eval(dataset, args.batchsize)
        #
        data10k, label10k = prepare_data_topn(dataloader, args.batchsize, args.num_grams)
        print(dataset, len(dataloader))

        evaluate_propagations(net, dataloader, data10k, device, layers, layers_bn, layers0, params, [], startime,
                              args, datasets, dataset, epoch0, 'all',
                              save_targets=True, compute_features=True, compute_gram=True)

        for idx in range(len(ratios)):
            kernel0 = [torch.from_numpy(kernel[idx].astype(np.float32)).clone().to(device) \
                       for kernel in kernels0]
            net0 = rewrite_network(net0, kernel0, weightname)

            kernel1 = [torch.from_numpy(kernel[idx].astype(np.float32)).clone().to(device) \
                       for kernel in kernels1]

            print(ratios[idx], [k0.shape for k0 in kernel0], [k1.shape for k1 in kernel1])
            print(time.perf_counter()-startime)

            evaluate_propagations(net0, dataloader, data10k, device, layers, layers_bn, layers0, params, kernel1, startime,
                                  args, datasets, dataset, epoch0, f'{idx:0>3}', compute_convproject=True)

    #

    return


if __name__ == '__main__':
    main_analyzing()
