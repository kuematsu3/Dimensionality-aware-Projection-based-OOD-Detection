import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np

from analyzing import *
from utils import *


def compute_gram_matrix(model, net, layers, data, device, dataset, epoch0, imodel):
    net.eval()
    gram_all = []
    for idx0, inputs0 in enumerate(data):
        # print(idx0, len(inputs0))
        inputs0 = inputs0.to(device)
        with torch.no_grad():  logits0, features0 = partialprop(model, net, inputs0, 0, layers)
        features0 = [feature0.view(len(inputs0), -1) for feature0 in features0]

        gram0 = []
        for idx1, inputs1 in enumerate(data):
            inputs1 = inputs1.to(device)
            len1 = len(inputs1)
            with torch.no_grad():  logits1, features1 = partialprop(model, net, inputs1, 0, layers)
            features1 = [feature1.view(len(inputs1), -1) for feature1 in features1]
            inner = [torch.mm(feature0, feature1.t()) for feature0, feature1 in zip(features0, features1)]
            gram0.append([inner0.detach().to('cpu').clone().numpy() for inner0 in inner])

        gram1 = [np.concatenate([gram[i] for gram in gram0], 1) for i in range(len(layers))]
        gram_all.append(gram1)

    for idx, layer in enumerate(layers):
        gram = np.concatenate([gram0[idx] for gram0 in gram_all], 0)
        np.savez_compressed(f'./features/gram{epoch0-1:0>4}-{layer:0>2}_{imodel}_'+dataset, gram)

    return



def obtain_centrizer(len0):
    centrizer_all = np.ones(len0) / np.sqrt(len0)
    centrizer_all = np.diag(np.ones(len0)) - np.outer(centrizer_all, centrizer_all)

    return centrizer_all



def obtain_sims_layers(eigens, datasets, layers, epoch0, name0, testdataset, device, compute_all=False, eps=1e-6):
    for eigens_data, dataset in zip(eigens, datasets):
        if not compute_all and dataset!='traindata' and dataset!=testdataset:  continue

        for idx0, (eigen0, layer0) in enumerate(zip(eigens_data, layers)):
            val0 = eigen0[0].astype(np.float32)
            eig0 = [torch.from_numpy(val0).clone().to(device),
                    torch.from_numpy(eigen0[1].astype(np.float32)).clone().to(device)]

            cka = val0*val0 / np.sum(val0*val0);  cka = cka[cka > np.amax(cka)*eps]
            cca = np.ones(len(val0))
            reg = val0 / np.sum(val0);  reg = reg[reg > np.amax(reg)*eps]
            np.savez_compressed(f'./similarity/{name0}{epoch0-1:0>4}-{layer0:0>2}-{layer0:0>2}_'+dataset,
                                cka=cka, cca=cca, reg0=reg, reg1=reg)

            for idx1, (eigen1, layer1) in enumerate(zip(eigens_data, layers)):
                eig1 = [torch.from_numpy(eigen1[0].astype(np.float32)).clone().to(device),
                        torch.from_numpy(eigen1[1].astype(np.float32)).clone().to(device)]
                if idx1 <= idx0: continue
                cka, cca, reg0, reg1 = obtain_similarities(eig0[0], eig0[1], eig1[0], eig1[1], eps=eps)

                cka = cka.detach().to('cpu').clone().numpy()
                cca = cca.detach().to('cpu').clone().numpy()
                reg0 = reg0.detach().to('cpu').clone().numpy()
                reg1 = reg1.detach().to('cpu').clone().numpy()

                print(dataset, layer0, layer1, cka.shape, cca.shape, reg0.shape, reg1.shape)
                np.savez_compressed(f'./similarity/{name0}{epoch0-1:0>4}-{layer0:0>2}-{layer1:0>2}_'+dataset,
                                    cka=cka, cca=cca, reg0=reg0, reg1=reg1)
                np.savez_compressed(f'./similarity/{name0}{epoch0-1:0>4}-{layer1:0>2}-{layer0:0>2}_'+dataset,
                                    cka=cka, cca=cca, reg0=reg0, reg1=reg1)

    return





# better to be dim(val0) > dim(val1)
def obtain_similarities(val0, vec0, val1, vec1, eps=1e-6):
    prod01 = torch.mm(vec0.t(), vec1)
    cka = torch.mm(torch.diag(torch.sqrt(val0)), torch.mm(prod01, torch.diag(torch.sqrt(val1))))

    if len(val0) > len(val1):
        cka = torch.mm(cka.t(), cka)
    else:
        cka = torch.mm(cka, cka.t())

    s_cka, vt = torch.linalg.eigh(cka / torch.sqrt(torch.sum(val0*val0) * torch.sum(val1*val1)))
    s_cka = s_cka[s_cka > torch.amax(s_cka)*eps]


    u, s_cca, vt = torch.linalg.svd(prod01)
    s_cca = s_cca[s_cca > torch.amax(s_cca)*eps]


    reg1 = torch.mm(torch.mm(prod01, torch.diag(val1)), prod01.t())
    s_reg1, vt = torch.linalg.eigh(reg1 / torch.sum(val1))
    s_reg1 = s_reg1[s_reg1 > torch.amax(s_reg1)*eps]

    reg0 = torch.mm(torch.mm(prod01.t(), torch.diag(val0)), prod01)
    s_reg0, vt = torch.linalg.eigh(reg0 / torch.sum(val0))
    s_reg0 = s_reg0[s_reg0 > torch.amax(s_reg0)*eps]


    return s_cka, s_cca, s_reg0, s_reg1



# name0 = 'gram-all-val'
def obtain_eigens(centrizer, centrizer1, device, layers,
                  datasets, name0, epoch0, testdataset, compute_all=False, eps=1e-6):
    centrizer = torch.from_numpy(centrizer.astype(np.float32)).clone().to(device)
    centrizer1 = torch.from_numpy(centrizer1.astype(np.float32)).clone().to(device)
    eigens = []
    for dataset in datasets:
        if not compute_all and dataset!='traindata' and dataset!=testdataset:  continue
        if dataset=='traindata':
            centr = centrizer
        elif dataset==testdataset:
            centr = centrizer1

        eigen = []
        for layer in layers:
            gram_np = np.load(f'./features/gram{epoch0-1:0>4}-{layer:0>2}_all_{dataset}.npz')['arr_0']

            tmp = torch.from_numpy(gram_np).clone().to(device)
            val0, vec0 = obtain_eigens_centrized_gram(tmp, centr, eps=eps)
            val0 = val0.detach().to('cpu').clone().numpy()
            vec0 = vec0.detach().to('cpu').clone().numpy()
            eigen.append([val0, vec0])
            print(dataset, layer, ' diagonalized: ', val0.shape, vec0.shape)
            np.savez_compressed(f'./similarity/{name0}{epoch0-1:0>4}-{layer:0>2}_'+dataset, val0)
        eigens.append(eigen)

    return eigens


def obtain_eigens_centrized_gram(gram, centrizer, eps=1e-6):
    gramc = torch.mm(torch.mm(centrizer, gram), centrizer)
    eigval, eigvec = torch.linalg.eigh(gramc)
    argsorts = torch.argsort(eigval, descending=True)

    eigval = eigval[argsorts]
    eigvec = eigvec[:, argsorts]
    bools = (eigval > eigval[0]*eps)

    return eigval[bools], eigvec[:, bools]

                                                


def main_kernels():
    startime = time.perf_counter()
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    epoch0 = args.start_epoch + 1
    nc = args.num_classes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weightname0, weightname = get_weight_dict_name(args.model_name)
    layers = [idx for idx in range(len(weightname))]
    datasets = ['traindata', 'cifar10', 'svhn', 'mnist', 'cifar100'] # second one must be in-distribution
    print('parameters prepared: ', time.perf_counter()-startime)

    c_all = obtain_centrizer(args.num_stability)
    print('class prepared: ', time.perf_counter()-startime)
    if not os.path.isdir('similarity'):  os.mkdir('similarity')

    eigens = obtain_eigens(c_all, c_all, device, layers, datasets,
                           'gram-all-val', epoch0, args.traindata, compute_all=True) # [dataset, layer, val/vec]
    print('centered gram matrices diagonalized: ', time.perf_counter()-startime)
    obtain_sims_layers(eigens, datasets, layers, epoch0, 'sim-all', args.traindata, device, compute_all=True)
    print('similarity between layers diagonalized: ', time.perf_counter()-startime)

    return



if __name__ == '__main__':
    main_kernels()
