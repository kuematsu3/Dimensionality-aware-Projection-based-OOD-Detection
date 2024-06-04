# just for certain model and certain random seed

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix

import sys, os, argparse
import time, math




def evaluate_performance(dir0, name0, datasets, traindata, scores, color_dataset, cm, layers_custom,
                         tpr_picked=[0.9, 0.95],
                         plt_roc=False, plt_pr=False, plt_nr=False, save_histogram=False, save_cumulative=False):
    idx_test = datasets.index(traindata)

    results = []
    for dataset, score in zip(datasets, scores):
        if dataset=='traindata' or dataset==traindata: continue
        rates = make_positive_rates(scores[idx_test], score) # fpr, tpr, ppv, npv, len0

        auroc = sklearn.metrics.auc(rates[0], rates[1])
        aupri = sklearn.metrics.auc(rates[1][1:], rates[2])
        aunri = sklearn.metrics.auc(rates[1][:len(rates[1])-1], rates[3])
        aupro = sklearn.metrics.auc(1.0-rates[0][:len(rates[0])-1], rates[3])
        aunro = sklearn.metrics.auc(1.0-rates[0][1:], rates[2])
        # auroc, aupr, tnr95 (TNR=1-FPR at TPR = 95%. detection rate of ID at 95% of OOD samples detected)
        # tnr=1-fpr at tpr=0.95 <-> fpr at fnr=1-tpr=0.95
        # tpr at tnr=1-fpr=0.95 <-> fnr=1-tpr at fpr=0.95

        eps = 1e-6
        fpr_thresh = []
        for rate0 in tpr_picked:
            bool0 = (rates[1] > (rate0 - eps)) & (rates[1] < (rate0 + eps))
            bool1 = (rates[1] > (1.0-rate0 - eps)) & (rates[1] < (1.0-rate0 + eps))
            bool2 = (rates[0] > (1.0-rate0 - eps)) & (rates[0] < (1.0-rate0 + eps))
            bool3 = (rates[0] > (rate0 - eps)) & (rates[0] < (rate0 + eps))
            fpr_thresh.append(1.0-np.mean(rates[0][bool0]))
            fpr_thresh.append(np.mean(rates[0][bool1]))
            fpr_thresh.append(np.mean(rates[1][bool2]))
            fpr_thresh.append(1.0-np.mean(rates[1][bool3]))

        results.append([auroc, aupri, aunri, aupro, aunro] + fpr_thresh)

    # print(len(results), [len(res) for res in results])
    # print([[r0.shape for r0 in res] for res in results])
    return np.array(results, dtype=np.float32)


def make_positive_rates(test, vals):
    if len(test.reshape(-1)) == len(vals.reshape(-1)):
        vals1 = vals.reshape(-1)
    else:
        vals1 = np.random.choice(vals.reshape(-1), len(test.reshape(-1)))

    arrays = np.concatenate([test.reshape(-1), vals1])
    boolin = np.full_like(arrays, False, dtype=bool)
    boolin[:len(test.reshape(-1))] = True

    argsorts = np.argsort(arrays)[::-1]
    boolin = boolin[argsorts]
    len0 = len(arrays)
    arr0 = np.arange(len0)

    ### positive == in distribution
    tp = np.cumsum(boolin)
    fp = np.cumsum(~boolin)
    tn = len(vals1) - fp
    fn = len(test.reshape(-1)) - tp
    npp = 1 + arr0 # number of positive prediction
    nnp = len0 - arr0 # number of negative prediction

    tpr = np.zeros(len0+1)
    fpr = np.zeros(len0+1)
    tpr[1:len0+1] = tp / len(test.reshape(-1))
    fpr[1:len0+1] = fp / len(vals1)
    ppv = tp / npp # corresponding to tpr[1:len(arrays)+1]
    npv = tn / nnp # corresponding to tpr[0:len(arrays)]

    return [fpr, tpr, ppv, npv, len0]

def investigate_outputs(dir0, args, datasets, layers, layers0, rates, ndims, epoch0,
                        color_layer, color_dataset, cm, layers_custom,
                        eps=1e-7, save_histogram=False, save_cumulative=False):
    print('investigate_outputs: ')
    targets = [np.load(dir0+f'outputs/labels{epoch0-1:0>4}_{dataset}.npz')['arr_0'] for dataset in datasets]

    idx_train = datasets.index('traindata')
    idx_test = datasets.index(args.traindata)
    accuracy, detections = [], []
    for idx in range(len(rates)+1):
        probs, logits, acc0 = [], [], []
        name0 = f'{idx-1:0>3}'
        if idx==0:  name0 = 'all'

        for idx_data, dataset in enumerate(datasets):
            logit = np.load(dir0+f'outputs/logits{epoch0-1:0>4}_{name0}_{dataset}.npz')['arr_0']
            if dataset == 'traindata' or dataset == args.traindata:
                inferred = np.argmax(logit, axis=1)
                acc0.append(np.count_nonzero((targets[idx_data]==inferred)) / len(inferred))
            logits.append(logit)
            prob = np.exp(logit-np.amax(logit, axis=1, keepdims=True))
            prob = np.amax(prob, axis=1) / np.sum(prob, axis=1)
            probs.append(-np.log(1.0-prob+eps))
        #
        accuracy.append(acc0)
        results = evaluate_performance(dir0, f'prob-{name0}', datasets, args.traindata, probs,
                                       color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
        detections.append(results)
        # print('   ', name0, acc0, results)
    #
    accuracy = np.array(accuracy, dtype=np.float32)
    detections = np.array(detections, dtype=np.float32)
    np.savez_compressed(dir0+'extract/output-scores',
                        accuracy=accuracy, detection=detections, ndims=ndims, rates=rates)

    return


def investigate_mahalanobis(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom):
    print('investigate_mahalanobis: ')
    idx_train = datasets.index('traindata')
    idx_test = datasets.index(args.traindata)
    imodel = 'all'

    sigmas = []
    for layer in layers:
        npz = np.load(dir0+f'features/sigma{epoch0-1:0>4}-{layer:0>2}_{imodel}_traindata.npz')
        sigmas.append([npz['tied']])

    name0 = ['tied']
    if not os.path.isdir(dir0+'extract/features'):  os.makedirs(dir0+'extract/features')


    for i in range(len(name0)):
        sigma0 = [sigma[i] for sigma in sigmas]
        spectrum = np.array([np.amax(sigma, axis=-1) for sigma in sigma0]).reshape(len(layers), -1)
        frobenius = np.array([np.sqrt(np.sum(sigma*sigma, axis=-1)) for sigma in sigma0]).reshape(len(layers), -1)
        trace = np.array([np.sum(sigma, axis=-1) for sigma in sigma0]).reshape(len(layers), -1)
        dims = np.array([len(sigma) for sigma in sigma0]).reshape(len(layers), -1)
        np.savez_compressed(dir0+f'extract/features/sigma{epoch0-1:0>4}-{name0[i]}-norm_{imodel}_traindata',
                            spectrum=spectrum, frobenius=frobenius, trace=trace, dims=dims)



    mahalanobis = []
    for dataset in datasets:
        mdist = []
        for layer in layers:
            mahal = []
            for iname in name0:
                iname0 = iname.replace('-', '')
                npz = np.load(dir0+f'features/mahalanobis-{iname0}{epoch0-1:0>4}-{layer:0>2}_{imodel}_{dataset}.npz')
                bools = npz['dims_exp'] <= npz['dims_eps'][-1]
                mahal.append([npz['distance_exp'][:, bools], npz['dims_exp'][bools]])
            mdist.append(mahal)
        mahalanobis.append(mdist)


    # dim0 = np.array([9, 16, 31, 65, 124, 272, 539, 1070, 2131, 4249], dtype=np.int32)
    if not os.path.isdir(dir0+'extract/mahalanobis'):  os.makedirs(dir0+'extract/mahalanobis')
    dim0 = np.array([-1], dtype=np.int32)
    for idx_cov, iname in enumerate(name0):
        for idx_layer, layer in enumerate(layers):
            mdist = [mahal[idx_layer][idx_cov] for mahal in mahalanobis]
            len0 = min([mahal[0].shape[-1] for mahal in mdist])
            xaxis = [mahal[1][:len0] for mahal in mdist]
            yaxis = [(mahal[0][:,:len0] / (1+mahal[1][:len0]))**(1.0/3.0) for mahal in mdist]
            auroc = []
            for idx_length in range(len0):
                scores = [score[:, idx_length] for score in yaxis]
                idim = xaxis[0][idx_length]
                save_histogram, save_cumulative = (idim in dim0), (idim in dim0)
                results = evaluate_performance(dir0+'extract/', f'mahal{layer:0>2}-{idim:0>4}', datasets,
                                               args.traindata, scores, color_dataset, cm, layers_custom,
                                               tpr_picked=[0.9, 0.95],
                                               save_histogram=save_histogram, save_cumulative=save_cumulative)
                auroc.append(results)

            np.savez_compressed(dir0+'extract/mahalanobis/mahalanobis-scores_'+iname+f'_{layer:0>2}',
                                auroc=np.array(auroc), dims=xaxis[0])

    return


def investigate_projections(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom):
    print('investigate_projections: ')
    results = []
    for layer in layers:
        name0 = f'{epoch0-1:0>4}-{layer:0>2}'
        results0 = []

        norm00_all = [np.load(dir0+f'projections/norm00-{name0}_all_{dataset}.npz')['arr_0'] \
                      for dataset in datasets]
        results00_all = evaluate_performance(dir0, 'norm00-all', datasets, args.traindata,
                                             norm00_all, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
        norm0_all = [np.load(dir0+f'projections/norm0-{name0}_all_{dataset}.npz')['arr_0'] \
                     for dataset in datasets]
        results0_all = evaluate_performance(dir0, 'norm0-all', datasets, args.traindata,
                                            norm0_all, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
        results0.append([results00_all, results0_all])

        for idx in range(len(rates)):
            name1 = f'{idx:0>3}'
            norm1_all = [np.load(dir0+f'projections/norm1-{name0}_all-{name1}_{dataset}.npz')['arr_0'] \
                          for dataset in datasets]
            results1_all = evaluate_performance(dir0, 'norm1-all', datasets, args.traindata,
                                                norm1_all, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            normr_all = [norm1_all[i] / norm0_all[i] for i in range(len(datasets))]
            resultsr_all = evaluate_performance(dir0, 'normr-all', datasets, args.traindata,
                                                normr_all, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            results0.append([results1_all, resultsr_all])

            norm00_small = [np.load(dir0+f'projections/norm00-{name0}_{name1}_{dataset}.npz')['arr_0'] \
                            for dataset in datasets]
            results00_small = evaluate_performance(dir0, 'norm00-'+name1, datasets, args.traindata,
                                                   norm00_small, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            norm0_small = [np.load(dir0+f'projections/norm0-{name0}_{name1}_{dataset}.npz')['arr_0'] \
                           for dataset in datasets]
            results0_small = evaluate_performance(dir0, 'norm0-'+name1, datasets, args.traindata,
                                                  norm0_small, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            results0.append([results00_small, results0_small])

            norm1_small = [np.load(dir0+f'projections/norm1-{name0}_{name1}-{name1}_{dataset}.npz')['arr_0'] \
                           for dataset in datasets]
            results1_small = evaluate_performance(dir0, 'norm1-'+name1, datasets, args.traindata,
                                                  norm1_small, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            normr_small = [norm1_small[i] / norm0_small[i] for i in range(len(datasets))]
            resultsr_all = evaluate_performance(dir0, 'normr-'+name1, datasets, args.traindata,
                                                normr_all, color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95])
            results0.append([results1_small, resultsr_all])
            # print('   ', name1, results1_all, results1_small)

        results.append(results0)
    results = np.array(results, dtype=np.float32)
    np.savez_compressed(dir0+'extract/projection-scores', auc=results, rates=rates)

    return



def investigate_similarities(dir0, args, datasets, layers, layers0, epoch0,
                             color_layer, color_dataset, cm, layers_custom,
                             hist_interdataset=False, hist_interlayer=False):
    print('investigate_similarities: ')
    datasets_id = ['traindata', args.traindata]

    vals_all = [[np.load(dir0+f'similarity/gram-all-val{epoch0-1:0>4}-{layer:0>2}_{dataset}.npz')['arr_0'] \
                 for layer in layers] for dataset in datasets]

    for dataset, vals in zip(datasets, vals_all):
        spectrum = np.array([np.amax(val, axis=-1) for val in vals])
        frobenius = np.array([np.sqrt(np.sum(val*val, axis=-1)) for val in vals])
        trace = np.array([np.sum(val, axis=-1) for val in vals])
        dims = np.array([len(val) for val in vals]).reshape(len(layers), -1)
        np.savez_compressed(dir0+f'extract/features/gram-all-val{epoch0-1:0>4}-norm_{dataset}',
                            spectrum=spectrum, frobenius=frobenius, trace=trace, dims=dims)


    similarities = ['cka', 'cca', 'reg0', 'reg1']
    if not os.path.isdir(dir0+'extract/similarity'):  os.makedirs(dir0+'extract/similarity')

    # interlayer similarity
    npz = [[[np.load(dir0+f'similarity/sim-all{epoch0-1:0>4}-{layer0:0>2}-{layer1:0>2}_{dataset}.npz') \
             for layer1 in layers] for layer0 in layers] for dataset in datasets]
    sims_all = [[[ [npz2['cka'], npz2['cca'], npz2['reg0'], npz2['reg1']] for npz2 in npz1] \
                 for npz1 in npz0] for npz0 in npz]

    # data, layer0, layer1, sims
    for idx_sim, simname in enumerate(similarities):
        normsinf, norms2, norms1, norms0 = [], [], [], []
        for dataset, sall in zip(datasets, sims_all):
            normsinf.append([[np.amax(sim0[idx_sim], axis=-1) for sim0 in sim] for sim in sall])
            norms2.append([[np.sqrt(np.sum(sim0[idx_sim]*sim0[idx_sim], axis=-1)) for sim0 in sim] for sim in sall])
            norms1.append([[np.sum(sim0[idx_sim], axis=-1) for sim0 in sim] for sim in sall])
            norms0.append([[len(sim0[idx_sim]) for sim0 in sim] for sim in sall])
        np.savez_compressed(dir0+f'extract/similarity/{simname}-all-norm_interlayers',
                            spectrum=np.array(normsinf), frobenius=np.array(norms2),
                            trace=np.array(norms1), dims=np.array(norms0))

    return



def investigate_stabilities(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom,
                            eps=1e-7, hist_contraction=False, hist_sensitivity=False):
    print('investigate_stabilities: ')
    name0 = dir0 + f'projections/norm00-{epoch0-1:0>4}-'
    if not os.path.isdir(dir0+'extract/sensitivity'):  os.makedirs(dir0+'extract/sensitivity')
    idx_train = datasets.index('traindata')

    # noise sensitivity
    name0 = dir0 + f'projections/stability{epoch0-1:0>4}_'
    eta = 0.1
    xaxis = [layers0[idx_layer:]+[len(layers)] for idx_layer in range(len(layers0))]
    for idx in range(len(rates)+1):
        name1 = f'{idx-1:0>3}'
        if idx==0: name1 = 'all'

        quants = []
        for dataset0 in datasets:
            quants0 = []
            for idx_layer, layer in enumerate(layers0):
                name2 = f'{layer:0>2}-eta{eta:.3f}_{name1}_{dataset0}'
                diff = np.load(name0+name2+f'-{dataset0}.npz')['arr_0']
                quants0.append(diff[1:])
            quants.append(quants0)

        # quants = [original-dataset, input-layers0, output-layers0 (exept input layer), samples]
        # xaxis = [layers0[idx_layer:]+[len(layers)] for idx_layer in range(len(layers0))]
        results = []
        for idx_layer0, layer0 in enumerate(layers0):
            results0 = [np.full((len(datasets)-2, 13), 0.5)]
            stats0 = [np.full((len(datasets), 5), eta*eta)]
            for idx_layer1, layer1 in enumerate(xaxis[idx_layer0][1:]):
                name2 = f'{layer0:0>2}-{layer1:0>2}-eta{eta:.3f}_{name1}_isotropic'

                sensitivity = [quant[idx_layer0][idx_layer1] for quant in quants]
                results2 = evaluate_performance(dir0+'extract/stability/', f'sensitivity'+name2, datasets,
                                                args.traindata, [np.log(chi+eps) for chi in sensitivity],
                                                color_dataset, cm, layers_custom, tpr_picked=[0.9, 0.95],
                                                save_histogram=hist_sensitivity)
                results0.append(results2)
                stats0.append([[np.quantile(sens, 0.5, axis=-1),
                                np.quantile(sens, 0.31731051, axis=-1),
                                np.quantile(sens, 0.68268949, axis=-1),
                                np.quantile(sens, 0.04550026, axis=-1),
                                np.quantile(sens, 0.95449974, axis=-1)] for sens in sensitivity])
            results0 = np.array(results0)
            stats0 = np.array(stats0)
            np.savez_compressed(dir0+f'extract/sensitivity/sensitivity{layer0:0>2}-scores_{name1}', auc=results0, stats=stats0)

    return



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
        blocks = [2, 2, 2, 2]
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


def set_dims_model(modelname, nc):
    if modelname=='vgg13':
        dims = [27, 64, 128, 128,
                256, 256,
                512, 512,
                512, 512, nc]
    elif modelname=='vgg16':
        dims = [27, 64, 128, 128,
                256, 256, 256,
                512, 512, 512,
                512, 512, 512, nc]
    elif modelname=='resnet18': # block3, block 5, block7
        dims = [27, 64, 64, 64, 64,
                128, 128, 128, 128,
                256, 256, 256, 256,
                512, 512, 512, 512, nc]
    elif modelname=='resnet34': # block4, block8, block14
        dims = [27, 64, 64, 64, 64, 64, 64,
                128, 128, 128, 128, 128, 128, 128, 128,
                256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                512, 512, 512, 512, 512, 512, nc]
    #
    return dims


def set_layers_model(modelname):
    if modelname=='vgg13':
        layers = [i for i in range(11)]
        plotted = [0, 2, 4, 6, 8, 10]
    elif modelname=='vgg16':
        layers = [i for i in range(14)]
        plotted = [0, 2, 4, 7, 10, 13]
    elif modelname=='resnet18': # block3, block 5, block7
        layers = [0] + [i for i in range(1, 18, 2)]
        plotted = [0, 5, 9, 13, 17]
    elif modelname=='resnet34': # block4, block8, block14
        layers = [0] + [i for i in range(1, 34, 2)]
        plotted = [0, 7, 15, 21, 27, 33]
    #
    return layers, plotted


def parse_args(argv):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model-name', default='resnet18', help='model name')
    
    parser.add_argument('--traindata', default='cifar10', help='train dataset')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size for train and test')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1200, type=int, help='training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch for training')

    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in total')
    parser.add_argument('--num_grams', default=10000, type=int, help='number of data for gram matrices')
    parser.add_argument('--num_stability', default=10000, type=int, help='number of data for noise stability')

    parser.add_argument('--rndmseed', default=0, type=int, help='index for random seed')

    return parser.parse_args(argv)


def print_parametercheck(args):
    print('=== {} ==='.format(args.model_name))
    print('traindata: {:s}'.format(args.traindata))
    print('total epochs: {:d}'.format(args.epochs))
    print('start epochs: {:d}'.format(args.start_epoch))
    print('lr: {:f}'.format(args.lr))
    print('batch size: {:d}'.format(args.batchsize))

    return





def main():
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    dir0 = f'/home/user/workdir/{args.traindata}/{args.model_name}/rnd{args.rndmseed}/'
    epoch0 = args.start_epoch + 1
    datasets = ['traindata', 'cifar10', 'svhn', 'mnist', 'cifar100']
    color_dataset = ['black', 'darkviolet', 'red', 'green', 'blue']
    colorbasis = ['salmon', 'indianred', 'darkred',
                  'red', 'darkorange', 'orange', 'yellow', 'greenyellow', 'limegreen',
                  'aquamarine', 'cyan', 'deepskyblue', 'dodgerblue', 'blue',
                  'darkblue', 'darkviolet', 'magenta', 'violet']
    cm = None
    # cm = generate_cmap(colorbasis)
    color_layer = None

    weightname0, weightname = get_weight_dict_name(args.model_name)
    dim_layers = set_dims_model(args.model_name, args.num_classes)
    layers = [idx for idx in range(len(weightname))]
    layers0, layers_custom = set_layers_model(args.model_name)
    if not os.path.isdir(dir0+'extract'):  os.makedirs(dir0+'extract')
    if not os.path.isdir(dir0+'extract/features'):  os.makedirs(dir0+'extract/features')

    npz = np.load(dir0+f'svals{epoch0-1:0>4}.npz')
    svals = npz['svals'];  ndims = npz['ndims'];  rates = npz['ratios']
    # dir, name, legends, quantities (yaxis), colors. (xaxis = indices of yaxis)
    svals = [sval[:diml] for sval, diml in zip(svals, dim_layers)]
    print([sval.shape for sval in svals])
    spectrum = np.array([np.amax(sval, axis=-1) for sval in svals])
    frobenius = np.array([np.sqrt(np.sum(sval*sval, axis=-1)) for sval in svals])
    trace = np.array([np.sum(sval, axis=-1) for sval in svals])
    np.savez_compressed(dir0+f'extract/sval{epoch0-1:0>4}-norm',
                        spectrum=spectrum, frobenius=frobenius, trace=trace, dims=np.array(dim_layers))


    investigate_outputs(dir0, args, datasets, layers, layers0, rates, ndims, epoch0,
                        color_layer, color_dataset, cm, layers_custom)
    investigate_mahalanobis(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom)
    investigate_projections(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom)
    investigate_similarities(dir0, args, datasets, layers, layers0, epoch0,
                             color_layer, color_dataset, cm, layers_custom)
    investigate_stabilities(dir0, args, datasets, layers, layers0, rates, epoch0,
                            color_layer, color_dataset, cm, layers_custom)


    return


if __name__ == '__main__':
    main()
