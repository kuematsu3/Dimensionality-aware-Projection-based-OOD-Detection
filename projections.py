import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import sys, os

from utils import *

class net_eval(nn.Module):

    def __init__(self, inch, outch, ksize, pad, stride, is_conv):
        super(net_eval, self).__init__()
        if is_conv:
            self.layer0 = nn.Conv2d(inch, outch, kernel_size=ksize,
                                    stride=stride, padding=pad, bias=False)
        else:
            self.layer0 = nn.Linear(inch, outch, bias=False)

    def forward(self, x):
        out = self.layer0(x)
        return out.detach()


def get_norm0_k3p1s1(inputs):
    if inputs.dim() != 4:
        print('invalid dimension of inputs to compute norm: size = ', inputs.size())
        sys.exit()

    nb = inputs.size(0)
    nc = inputs.size(1)
    ny = inputs.size(2) - 1
    nx = inputs.size(3) - 1

    if nx > 1 and ny > 1:
        inputs_bulk = inputs[0:nb, 0:nc, 1:ny, 1:nx].reshape(nb, -1)

        inputs_edge0 = inputs[0:nb, 0:nc, 0, 1:nx].reshape(nb, -1)
        inputs_edge1 = inputs[0:nb, 0:nc, ny, 1:nx].reshape(nb, -1)
        inputs_edge2 = inputs[0:nb, 0:nc, 1:ny, 0].reshape(nb, -1)
        inputs_edge3 = inputs[0:nb, 0:nc, 1:ny, nx].reshape(nb, -1)

        inputs_vtx0 = inputs[0:nb, 0:nc, 0, 0].reshape(nb, -1)
        inputs_vtx1 = inputs[0:nb, 0:nc, ny, 0].reshape(nb, -1)
        inputs_vtx2 = inputs[0:nb, 0:nc, 0, nx].reshape(nb, -1)
        inputs_vtx3 = inputs[0:nb, 0:nc, ny, nx].reshape(nb, -1)

        norm_bulk = torch.sum(inputs_bulk*inputs_bulk, 1)
        norm_edge = torch.sum(inputs_edge0*inputs_edge0, 1) + torch.sum(inputs_edge1*inputs_edge1, 1) \
                  + torch.sum(inputs_edge2*inputs_edge2, 1) + torch.sum(inputs_edge3*inputs_edge3, 1)
        norm_vtx = torch.sum(inputs_vtx0*inputs_vtx0, 1) + torch.sum(inputs_vtx1*inputs_vtx1, 1) \
                 + torch.sum(inputs_vtx2*inputs_vtx2, 1) + torch.sum(inputs_edge3*inputs_edge3, 1)
        
        norm0 = 9*norm_bulk + 6*norm_edge + 4*norm_vtx
    elif nx == 1 and ny == 1:
        inputs0 = inputs.reshape(nb, -1)
        norm0 = 4*torch.sum(inputs0*inputs0, 1)
    else:
        print('insufficient size of inputs to compute norm: size = ', inputs.size())
        sys.exit()
    
    return norm0



def get_norm0_k3p1s2(inputs):
    if inputs.dim() != 4:
        print('invalid dimension of inputs to compute norm: size = ', inputs.size())
        sys.exit()

    nb = inputs.size(0)
    nc = inputs.size(1)
    ny = inputs.size(2) - 1
    nx = inputs.size(3) - 1

    if nx > 1 and ny > 1:
        inputs_bulk0 = inputs[0:nb, 0:nc, 1:ny:2, 1:nx:2].reshape(nb, -1)
        inputs_bulk1 = inputs[0:nb, 0:nc, 0:ny:2, 1:nx:2].reshape(nb, -1)
        inputs_bulk2 = inputs[0:nb, 0:nc, 1:ny:2, 0:nx:2].reshape(nb, -1)
        inputs_bulk3 = inputs[0:nb, 0:nc, 0:ny:2, 0:nx:2].reshape(nb, -1)

        inputs_edge0 = inputs[0:nb, 0:nc, ny, 1:nx:2].reshape(nb, -1)
        inputs_edge1 = inputs[0:nb, 0:nc, ny, 0:nx:2].reshape(nb, -1)
        inputs_edge2 = inputs[0:nb, 0:nc, 1:ny:2, nx].reshape(nb, -1)
        inputs_edge3 = inputs[0:nb, 0:nc, 0:ny:2, nx].reshape(nb, -1)

        inputs_vtx0 = inputs[0:nb, 0:nc, ny, nx].reshape(nb, -1)

        norm_4 = torch.sum(inputs_bulk0*inputs_bulk0, 1)
        norm_2 = torch.sum(inputs_bulk1*inputs_bulk1, 1) + torch.sum(inputs_bulk2*inputs_bulk2, 1) \
               + torch.sum(inputs_edge0*inputs_edge0, 1) + torch.sum(inputs_edge2*inputs_edge2, 1)
        norm_1 = torch.sum(inputs_bulk3*inputs_bulk3, 1) \
               + torch.sum(inputs_edge1*inputs_edge1, 1) + torch.sum(inputs_edge3*inputs_edge3, 1) \
               + torch.sum(inputs_vtx0*inputs_vtx0, 1)

        norm0 = 4*norm_4 + 2*norm_2 + norm_1
    else:
        print('insufficient size of inputs to compute norm: size = ', inputs.size())
        sys.exit()
    
    return norm0



def get_norm0(features, layers, params):
    norms00, norms0 = [], []
    for idx, layer in enumerate(layers):
        feature = features[idx]
        param = params[layer]
        nb = feature.size(0)

        viewed = feature.reshape(nb, -1)
        norm00 = torch.sum(viewed*viewed, 1)

        if param[5]:
            assert param[2]==3, 'invalid kernel size. only 3x3 available.'
            if param[4]==1:
                norm0 = get_norm0_k3p1s1(feature)
            elif param[4]==2:
                norm0 = get_norm0_k3p1s2(feature)
        else:
            norm0 = norm00

        norms00.append(norm00)
        norms0.append(norm0)

    return norms00, norms0



def projection_norms(layers, features, kernels, params, device):
    norms1 = []
    for idx, layer in enumerate(layers):
        kernel1 = kernels[layer]
        feature = features[idx]
        param = params[idx]
        nch = kernel1.size(0)

        net1 = net_eval(param[0], param[1], param[2], param[3], param[4], param[5])
        net1 = net1.to(device)
        if device == 'cuda':
            net1 = torch.nn.DataParallel(net1)
            cudnn.benchmark = True
        net1.eval()
        with torch.no_grad():
            net1.state_dict()['module.layer0.weight'][0:nch] = kernel1
            out = net1.module.layer0(feature)

        out0 = out.reshape(len(out), -1)
        norm1 = torch.sum(out0*out0, 1)

        norms1.append(norm1)


    return norms1




def compute_svd(net, weightname):

    kernel_svd = []
    shapes = []

    net.eval()
    with torch.no_grad():
        for convname in weightname:
            kernel0 = net.state_dict()[convname].clone()
            u, s, v = torch.svd(kernel0.view(kernel0.size(0), -1))
            kernel_svd.append([u, s, v])
            shapes.append(kernel0.size())


    return kernel_svd, shapes


def get_ndims(sval, ratios, smaller=True):
    s1 = sval.clone()
    # ndim = torch.count_nonzero(s1 > (s1[0]*r0))
    ndims = [int(torch.sum(s1 > (s1[0]*r0)).item()) for r0 in ratios]

    if smaller:
        ndims = [min(max(ndim, 1), len(s1)) for ndim in ndims]
    else:
        ndims = [min(ndim, len(s)-1) for ndim in ndims]

    return ndims


def get_kernels(usv, ndims, shape0, smaller=True, identity=False):
    s1 = usv[1].clone()

    kernels = []
    if smaller:
        for ndim in ndims:
            if ndim < len(s1): s1[ndim:] = 0.0
            if identity:  s1[:ndim] = 1.0
            kernel1 = torch.mm(torch.mm(usv[0], torch.diag(s1)), usv[2].t()).view(shape0)
            kernels.append(kernel1.detach().to('cpu').clone().numpy())
    else:
        for ndim in ndims:
            s1[:ndim] = 0.0
            if identity:  s1[ndim:] = 1.0
            kernel1 = torch.mm(torch.mm(usv[0], torch.diag(s1)), usv[2].t()).view(shape0)
            kernels.append(kernel1.detach().to('cpu').clone().numpy())

    return kernels



def get_weight_dict_name(model):
    if model == 'vgg13':
        blocks = [2, 2, 2, 2, 2]
        nconv = 1
    elif model == 'vgg16':
        blocks = [2, 2, 3, 3, 3]
        nconv = 1
    elif model == 'resnet18':
        blocks = [2, 2, 2, 2]
        nconv = 2
    elif model == 'resnet34':
        blocks = [3, 4, 6, 3]
        nconv = 2

    idx, weightname = 0, []
    if 'resnet' in model: weightname.append('module.conv1.weight')
    for istage, nblock in enumerate(blocks):
        for iblock in range(nblock):
            for iconv in range(nconv):
                if 'resnet' in model:
                    weightname.append(f'module.layer{istage+1}.{iblock}.conv{iconv+1}.weight')
                elif 'vgg' in model:
                    weightname.append(f'module.features.{idx}.weight')
                idx += 3
        idx += 1
    if 'resnet' in model:
        weightname.append('module.fc.weight')
    elif 'vgg' in model:
        weightname.append('module.classifier.weight')

    weightname0 = []
    for wname in weightname:
        if 'conv2' in wname: continue
        weightname0.append(wname)

    return weightname0, weightname



def get_parameters(net, weightname, ratios):
    kernel_svd, shapes = compute_svd(net, weightname)
    svals = [usv[1].detach().to('cpu').clone().numpy() for usv in kernel_svd]
    ndims = [get_ndims(usv[1], ratios, smaller=True) for usv in kernel_svd]
    kernel_proj = [get_kernels(usv, ndim, shape0, smaller=True, identity=True) \
                   for usv, ndim, shape0 in zip(kernel_svd, ndims, shapes)]
    kernel_compress = [get_kernels(usv, ndim, shape0, smaller=True, identity=False) \
                       for usv, ndim, shape0 in zip(kernel_svd, ndims, shapes)]

    return ndims, svals, kernel_proj, kernel_compress


def rewrite_network(net, kernels, weightname):
    net.eval()
    with torch.no_grad():
        for kernel0, convname in zip(kernels, weightname):
            nch = kernel0.size(0)
            net.state_dict()[convname][0:nch] = kernel0
        #
    #
    return net


def conv_projection(model, net, kernel1, loader, device, params, 
                    layers, epoch0, dataset, imodel, ieval):
    norms1 = []
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        len0 = len(inputs)
        with torch.no_grad():  logits, features = partialprop(model, net, inputs, 0, layers)

        norm1 = projection_norms(layers, features, kernel1, params, device)
        norms1.append([norm.detach().to('cpu').clone().numpy() for norm in norm1])


    for idx, layer in enumerate(layers):
        if not os.path.isdir('projections'):  os.mkdir('projections')
        norm1 = np.concatenate([norms[idx] for norms in norms1], 0)
        np.savez_compressed(f'./projections/norm1-{epoch0-1:0>4}-{layer:0>2}_{imodel}-{ieval}_'+dataset, norm1)
                                                    
    return 


