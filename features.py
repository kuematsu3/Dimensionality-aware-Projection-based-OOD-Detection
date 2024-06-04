import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys, os, glob, argparse, math, time
import numpy as np

from utils import *
from projections import *
from kernels import *
from analyzing import *


def avg_feature(feature, param, device): # feature.shape = (batchsize, num_channel, ysize, xsize)
    inch, outch, ksize, pad, stride, is_conv = param
    if not is_conv:
        return [feature]
    else:
        return [torch.sum(feature, dim=[2, 3])]
#


def compute_eigens(cov, eps=1e-6):
    vals, vecs = torch.linalg.eigh(cov)

    argsorts = torch.argsort(vals, descending=True)
    vals = vals[argsorts]
    vecs = vecs[:,argsorts]

    return vals, vecs


def get_cov(features, eps=1e-6):
    mu = torch.mean(features, 0, keepdim=True)
    diff = features - mu
    mu = mu.reshape(-1)
    cov = torch.mm(diff.t(), diff) / len(diff)
    vals, vecs = compute_eigens(cov, eps=eps)

    return mu, cov, vals, vecs


def set_save_dimensions_exp(len0, srate=10**(0.05)):
    init = int(1.0/(srate-1.0)) + 1
    dims = [i for i in range(min(init+1, len0))]
    for i in range(len0):
        init = int(init*srate)
        if init >= len0: break
        dims.append(init)

    return np.array(dims, dtype=np.int32)


def set_save_dimensions_eps(sigmar,
                            epsilons=[5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2,
                                      5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4,
                                      5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]):
    dims = [np.count_nonzero(sigmar > eps)-1 for eps in epsilons]

    return np.array(dims, dtype=np.int32)




def compute_eachlayers_np(layers, features_np, covs, device, name0, epoch0, imodel, dataset,
                          save_result=False, batchsize=5000):
    argmins_all, mdists_all = [], []
    for layer, feature_np, cov in zip(layers, features_np, covs):
        feature = torch.from_numpy(feature_np.astype(np.float32)).clone().to(device)
        feature = feature.reshape(len(feature), -1)

        mdists, argmins = [], []
        for idx in range((len(feature)-1) // batchsize + 1):
            feature0 = feature[idx*batchsize : min(len(feature),(idx+1)*batchsize)]
            mdist0, argmin0 = get_mahal_classmin(cov, feature0, device)
            mdists.append(mdist0);  argmins.append(argmin0)
        mdists = np.concatenate(mdists, 0);  argmins = np.concatenate(argmins, 0)

        digma = cov[1].ndim
        nc = len(cov[0])
        if digma==1: # tied
            sigmar = cov[1] / np.amax(cov[1])
        else: # class-wise
            sigmar = cov[1] / np.amax(cov[1], axis=1, keepdims=True)
            sigmar = np.prod(np.power(sigmar, 1.0/nc), 0)

        mdists_all.append(mdists)
        argmins_all.append(argmins)
        if save_result:
            dims_exp = set_save_dimensions_exp(feature.shape[1])
            # sigmar = sigmar.detach().to('cpu').clone().numpy()
            dims_eps = set_save_dimensions_eps(sigmar)
            print(name0, cov[0].shape, cov[1].shape, cov[2].shape, mdists.shape, argmins.shape, dims_eps[2::3])
            np.savez_compressed(f'./features/mahalanobis-{name0}{epoch0-1:0>4}-{layer:0>2}_{imodel}_'+dataset,
                                distance_exp=mdists[:, dims_exp], distance_eps=mdists[:, dims_eps],
                                inferred_exp=argmins[:, dims_exp], inferred_eps=argmins[:, dims_eps],
                                dims_exp=dims_exp, dims_eps=dims_eps)

    return mdists_all, argmins_all




# invals = 1.0 / vals.reshape(1,-1)
def get_mahal(features, mu, sigma, vecs):
    invals = 1.0 / sigma
    dots = torch.mm(features - mu, vecs)
    mdist = torch.cumsum(dots * dots * invals, axis=1)
    return mdist.detach().to('cpu').clone().numpy()



# invals = 1.0 / vals.reshape(1,-1)
def get_mahal_classmin(cov, feature, device):
    dimu = cov[0].ndim
    digma = cov[1].ndim
    nc = len(cov[0])

    if digma==1: # tied
        mdists = [get_mahal(feature,
                            torch.from_numpy(cov[0][ic].astype(np.float32)).clone().to(device).reshape(1,-1),
                            torch.from_numpy(cov[1].astype(np.float32)).clone().to(device).reshape(1,-1),
                            torch.from_numpy(cov[2].astype(np.float32)).clone().to(device)) \
                  for ic in range(nc)]
    else: # class-wise
        mdists = [get_mahal(feature,
                            torch.from_numpy(cov[0][ic].astype(np.float32)).clone().to(device).reshape(1,-1),
                            torch.from_numpy(cov[1][ic].astype(np.float32)).clone().to(device).reshape(1,-1),
                            torch.from_numpy(cov[2][ic].astype(np.float32)).clone().to(device)) \
                  for ic in range(nc)]

    mdists = np.array(mdists)
    argmins = np.argmin(mdists, axis=0)
    mdists = np.amin(mdists, axis=0)


    return mdists, argmins



def compute_covariances_np(layers, features_np, targets, nc, epoch0, imodel, name0, device, save_eigvecs=False):
    cov_tied_layers = []
    for layer, feature_np in zip(layers, features_np):
        feature = torch.from_numpy(feature_np.astype(np.float32)).clone().to(device)
        len0 = len(feature)
        dim0 = feature.numel() // len0
        # dim0 = feature.size // len0
        inv0 = 1.0 / (len0 * nc)

        mu_cls = []
        cov_tied = torch.zeros((dim0, dim0), device=device)
        for ic in range(nc):
            feature0 = feature[targets==ic]
            len1 = len(feature0)
            mu, cov, sigma, proj = get_cov(feature0.reshape(len1, -1))
            cov_tied += (cov*len1*inv0)
            mu_cls.append(mu.detach().to('cpu').clone().numpy())

        mu_cls = np.array(mu_cls)

        sigma_tied, proj_tied = compute_eigens(cov_tied)
        sigma_tied = sigma_tied.detach().to('cpu').clone().numpy()
        proj_tied = proj_tied.detach().to('cpu').clone().numpy()
        cov_tied_layers.append([mu_cls, sigma_tied, proj_tied])

        if save_eigvecs:
            np.savez_compressed(f'./features/{name0}-tied-{epoch0-1:0>4}-{layer:0>2}_{imodel}_traindata',
                                mu=mu_cls, vals=sigma_tied, vecs=proj_tied)

    return cov_tied_layers




def main_features():
    startime = time.perf_counter()
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_lr = args.lr * args.batchsize / 64.0
    print('device: ', device, time.perf_counter()-startime)

    trainloader, _, num_iteration = prepare_data_eval(args.traindata, args.batchsize)
    net, optimizer, scheduler, epoch0 = set_model(args, device, num_iteration, args.num_classes)
    params = set_param(args.model_name, args.num_classes)
    print('model prepared', time.perf_counter()-startime)

    datasets = ['traindata', 'cifar10', 'svhn', 'mnist', 'cifar100']
    weightname0, weightname = get_weight_dict_name(args.model_name)
    layers = [idx for idx in range(len(weightname))]
    print('parameters prepared', time.perf_counter()-startime)

    for layer in layers:
        targets, features = test(args.model_name, net, trainloader, device, [layer], params,
                                 'eval', epoch0, 'all', args.num_classes, compute_features=True)
        print(layer, len(features), features[0].shape)
        print('train feature acquired', time.perf_counter()-startime)

        cov_tied = compute_covariances_np([layer], features, targets, args.num_classes,
                                          epoch0, 'all', 'gauss', device)
        print('covariance acquired', time.perf_counter()-startime)

        for dataset in datasets:
            if dataset=='traindata':
                dataloader = trainloader
            else:
                _, dataloader, _ = prepare_data_eval(dataset, args.batchsize)

            _, features = test(args.model_name, net, dataloader, device, [layer], params,
                               'eval', epoch0, 'all', args.num_classes, compute_features=True)
            print(len(features), features[0].shape)
            print('feature of '+dataset+' acquired', time.perf_counter()-startime)

            mahal_tied, infer_tied = compute_eachlayers_np([layer], features, cov_tied, device,
                                                           'tied', epoch0, 'all', dataset, save_result=True)

    return


if __name__ == '__main__':
    main_features()
