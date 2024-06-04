# just for certain model and certain random seed

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams, cycler, cm
from matplotlib.lines import Line2D
from matplotlib import colors as mplcolors

import sys, os, argparse
import time, math


def plt_norms(dir0, matname, normname, statname, names, layers, center, width, colors, pltext, lowdiml, model):
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.15, right=1.00, bottom=0.12, top=0.9)
    plt.yscale('log')
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('stable rank', fontsize=20)
    ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
    ymax = max([np.amax(center0+width0) for center0, width0 in zip(center,width)])
    ax1.set_ylim([1, ymax])
    for center0, width0, iname, color0 in zip(center, width, names, colors):
        for j in range(center0.shape[-1]):
            if j==0:
                if width0.ndim==3:
                    plt.errorbar(layers, center0[:,j], yerr=width0[:,:,j],
                                 capsize=5, label=iname, color=color0, linewidth=0.5)
                elif width0.ndim==2:
                    plt.errorbar(layers, center0[:,j], yerr=width0[:,j],
                                 capsize=5, label=iname, color=color0, linewidth=0.5)
            else:
                if width0.ndim==3:
                    plt.errorbar(layers, center0[:,j], yerr=width0[:,:,j],
                                 capsize=5, color=color0, linewidth=0.5)
                elif width0.ndim==2:
                    plt.errorbar(layers, center0[:,j], yerr=width0[:,j],
                                 capsize=5, color=color0, linewidth=0.5)

                    #
    ax1.legend(bbox_to_anchor=(0,0.6), loc='lower left', fontsize=16)
    fig.savefig(dir0+matname+"_"+normname+'-'+statname+".pdf")
    return



def plt_weight_norms(dir0, matname, normname, statname, names, layers, center, width, colors, pltext, lowdiml, model):
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.15, right=1.00, bottom=0.12, top=0.9)
    plt.yscale('log')
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('stable rank', fontsize=20)
    ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
    ymax = max([np.amax(center0+width0) for center0, width0 in zip(center,width)])
    ax1.set_ylim([1, ymax])
    for center0, width0, iname, color0 in zip(center, width, names, colors):
        plt.errorbar(layers, center0, yerr=width0,
                     capsize=5, label=iname, color=color0, linewidth=0.5)
    #
    ax1.legend(bbox_to_anchor=(0,0.65), loc='lower left', fontsize=16)
    fig.savefig(dir0+matname+"_"+normname+'-'+statname+".pdf")
    return



def plt_inference(dir0, datasets, classes, center, width, colors, model, pltext, ymax=None):
    inv_nc = 1.0 / len(classes)

    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.11, right=1.00, bottom=0.12, top=0.92)
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('class', fontsize=20)
    ax1.set_ylabel('inferred samples', fontsize=20)
    ax1.axhline(y=inv_nc, xmin=-0.5, xmax=len(classes)-0.5, linestyle='--', linewidth=0.5, color='gray')
    if ymax==None:  ymax = max([np.amax(center0+width0) for center0, width0 in zip(center,width)])
    ax1.set_ylim([0, ymax])
    for center0, width0, iname, color0 in zip(center, width, datasets, colors):
        plt.errorbar(classes, center0, yerr=width0,
                     capsize=5, label=iname, color=color0, linewidth=0.5)
    #
    ax1.legend(bbox_to_anchor=(1.0, 1.00), loc='upper right', fontsize=16)
    fig.savefig(dir0+"inference.pdf")
    return



def load_norms(dir0, paths, name0, lowquant=0.25, highquant=0.75):
    normi, norm2, norm1, norm0 = [], [], [], []
    for path0 in paths:
        npz = np.load(dir0+path0+f'{name0}.npz')
        norm0.append(npz['dims'])
        norm1.append(npz['trace'])
        norm2.append(npz['frobenius'])
        normi.append(npz['spectrum'])

    normi = np.array(normi);  norm2 = np.array(norm2);  norm1 = np.array(norm1);  norm0 = np.array(norm0)
    rank1 = norm1 / normi;  rank2 = norm2 / normi
    rank2 = rank2 * rank2
    rankr1 = rank1 / norm0;  rankr2 = rank2 / norm0

    normi_stat = [np.mean(normi, axis=0), np.std(normi, axis=0), np.quantile(normi, 0.5, axis=0),
                  np.quantile(normi, lowquant, axis=0), np.quantile(normi, highquant, axis=0)]
    norm2_stat = [np.mean(norm2, axis=0), np.std(norm2, axis=0), np.quantile(norm2, 0.5, axis=0),
                  np.quantile(norm2, lowquant, axis=0), np.quantile(norm2, highquant, axis=0)]
    norm1_stat = [np.mean(norm1, axis=0), np.std(norm1, axis=0), np.quantile(norm1, 0.5, axis=0),
                  np.quantile(norm1, lowquant, axis=0), np.quantile(norm1, highquant, axis=0)]

    rank2_stat = [np.mean(rank2, axis=0), np.std(rank2, axis=0), np.quantile(rank2, 0.5, axis=0),
                  np.quantile(rank2, lowquant, axis=0), np.quantile(rank2, highquant, axis=0)]
    rank1_stat = [np.mean(rank1, axis=0), np.std(rank1, axis=0), np.quantile(rank1, 0.5, axis=0),
                  np.quantile(rank1, lowquant, axis=0), np.quantile(rank1, highquant, axis=0)]

    rankr2_stat = [np.mean(rankr2, axis=0), np.std(rankr2, axis=0), np.quantile(rankr2, 0.5, axis=0),
                   np.quantile(rankr2, lowquant, axis=0), np.quantile(rankr2, highquant, axis=0)]
    rankr1_stat = [np.mean(rankr1, axis=0), np.std(rankr1, axis=0), np.quantile(rankr1, 0.5, axis=0),
                   np.quantile(rankr1, lowquant, axis=0), np.quantile(rankr1, highquant, axis=0)]

    stats = [np.array(normi_stat), np.array(norm2_stat), np.array(norm1_stat),
             np.array(rank2_stat), np.array(rank1_stat), np.array(rankr2_stat), np.array(rankr1_stat)]

    return np.array(stats)




def average_covweight(dir0, model, seeds, paths, epoch0, layers, pltext, lowdiml, lowquant=0.25, highquant=0.75):
    weights = load_norms(dir0, paths, f'sval{epoch0:0>4}-norm')
    tiedcovs = load_norms(dir0, paths, f'features/sigma{epoch0:0>4}-tied-norm_all_traindata')

    center = [tiedcovs[3,2,:,0],  weights[3,2]];
    width = [np.abs(np.array([tiedcovs[3,2,:,0], tiedcovs[3,2,:,0]]) - tiedcovs[3,3:5,:,0]),
             np.abs(np.array([weights[3,2], weights[3,2]]) - weights[3,3:5])]
    print('weight:', weights[3,2])
    plt_weight_norms(dir0, 'cov-weight', 'srank', 'medi', ['tied cov', 'weight'],
                     layers, center, width, ['red', 'blue'], pltext, lowdiml, model)

    return




def plt_dimension_mahalanobis(dir0, name0, paths, seeds, layers, results,
                              dataref, data0, idx_score=0, scorename='AUROC'):
    for idx, (irnd, path0) in enumerate(zip(seeds, paths)):

        fig, ax1 = plt.subplots()
        ax1.set_xlim([1, 5000])
        ax1.set_xscale('log')
        ax1.set_ylim([0.5, 1])
        for layer, result in zip(layers, results):
            ax1.plot(result[idx][1], 1.0-result[idx][0][:, dataref, idx_score], label=f'{layer:0>2}')
        ax1.legend()
        fig.savefig(dir0+path0+f"{name0}_{scorename}_{data0}.pdf")
        plt.close('all')

    return



def plt_mahalanobis_auc_layers(dir0, name_score, name_stat, names, datasets,
                               layers, center, width, colors, lowdiml, model, pltext=''):

    for iname, center0, width0 in zip(names, center, width):
        fig, ax1 = plt.subplots()
        plt.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.9)
        ax1.set_ylim([0.5, 1])
        ax1.set_title(pltext+' '+model, fontsize=24)
        ax1.set_xlabel('layer', fontsize=20)
        ax1.set_ylabel(name_score, fontsize=20)
        ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
        for idx_data, dataset in enumerate(datasets):
            ax1.errorbar(layers, 1.0-center0[:, idx_data], yerr=width0[:, idx_data],
                         capsize=5, label=dataset, color=colors[idx_data], linewidth=0.5)
        ax1.legend(bbox_to_anchor=(1,0), loc='lower right', fontsize=16)
        fig.savefig(dir0+f"{iname}_{name_score}-{name_stat}.pdf")
        plt.close('all')

    return


def plt_projections_auc_layers(dir0, name_score, name_stat, names, pltexts, 
                               datasets, layers, center, width, colors, lowdiml, model, diff0=1.0):

    for iname, center0, width0, pltext in zip(names, center, width, pltexts):
        fig, ax1 = plt.subplots()
        plt.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.9)
        ax1.set_ylim([0.5, 1])
        ax1.set_title(pltext+' '+model, fontsize=24)
        ax1.set_xlabel('layer', fontsize=20)
        ax1.set_ylabel(name_score, fontsize=20)
        ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
        for idx_data, dataset in enumerate(datasets):
            ax1.errorbar(layers, np.abs(diff0-center0[:, idx_data]), yerr=width0[:, :, idx_data],
                         capsize=5, label=dataset, color=colors[idx_data], linewidth=0.5)
        ax1.legend(bbox_to_anchor=(0,0), loc='lower left', fontsize=16)
        fig.savefig(dir0+f"{iname}_{name_score}-{name_stat}.pdf")
        plt.close('all')

    return


def plt_similarity_interlayer(dir0, name0, statname, center, dataset, pltext, textpos, model, vrange, prange, vlabel):
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=prange[0], right=prange[1], bottom=prange[2], top=prange[3])
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('layer', fontsize=20)
    ax1.set_aspect('equal')

    cs = ax1.pcolor(center, cmap='inferno', norm=mplcolors.LogNorm(vmin=vrange[0], vmax=vrange[1]))

    cbar = fig.colorbar(cs)
    cbar.set_label(vlabel, fontsize=20)
    # ax1.text(textpos[0], textpos[1], pltext, fontsize=24)
    fig.savefig(dir0+f"{name0}-layers_{dataset}_{statname}.pdf")
    plt.close('all')

    return



def plt_similarity_penultimate(dir0, name0, statname, layers, center, width, datasets, colors, pltext, lowdiml, model, vrange):
    
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.9)
    ax1.set_ylim(vrange)
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_yscale('log')
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('CKA', fontsize=20)
    ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
    for idx_data, dataset in enumerate(datasets):
        if width.ndim==3:
            ax1.errorbar(layers, center[idx_data], yerr=width[:, idx_data],
                         capsize=5, label=dataset, color=colors[idx_data], linewidth=0.5)
    ax1.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right', fontsize=16)
    fig.savefig(dir0+f"{name0}_penultimate-{statname}.pdf")
    plt.close('all')
    
    return


def plt_sensitivity_stats(dir0, name0, legends, xaxis, center, width, colors, layers_custom, cm_mine,
                          lowdiml, pltext, model, ylim=None, plt_legend=False):
    # stable rank ratio
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.9)
    plt.yscale('log')
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('noise sensitivity', fontsize=20)
    ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
    ax1.axhline(y=1e-4, xmin=0.05, xmax=0.95, linestyle='--', linewidth=3, color='gray')
    if ylim != None:  ax1.set_ylim(ylim)
    for i in range(len(xaxis)):
        plt.errorbar(xaxis[i], center[i], yerr=width[i],
                     capsize=5, label=legends[i], color=colors[i], linewidth=0.5)
    if not plt_legend:
        tmp = [layer/layers_custom[-1] for layer in layers_custom]
        custom_lines = [Line2D([0], [0], color=cm_mine(tmp0), lw=0.5) for tmp0 in tmp]
        ax1.legend(custom_lines, layers_custom, bbox_to_anchor=(0.0, 0.0), loc='lower left', fontsize=16)
    else:
        ax1.legend(bbox_to_anchor=(0.0, 0.0), loc='lower left', fontsize=16)
    fig.savefig(dir0+name0+".pdf")
    plt.close('all')

    return


def plt_sensitivity_auc(dir0, name0, legends, xaxis, center, width, colors, layers_custom, cm_mine,
                          lowdiml, pltext, model, ylim=None, plt_legend=False):
    # stable rank ratio
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.9)
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('layer', fontsize=20)
    ax1.set_ylabel('AUROC', fontsize=20)
    ax1.axvline(x=lowdiml, ymin=0.05, ymax=0.95, linestyle='--', linewidth=3, color='gray')
    if ylim != None:  ax1.set_ylim(ylim)
    for i in range(len(xaxis)):
        plt.errorbar(xaxis[i], center[i], yerr=width[i],
                     capsize=5, label=legends[i], color=colors[i], linewidth=0.5)
    if not plt_legend:
        tmp = [layer/layers_custom[-1] for layer in layers_custom]
        custom_lines = [Line2D([0], [0], color=cm_mine(tmp0), lw=0.5) for tmp0 in tmp]
        ax1.legend(custom_lines, layers_custom, bbox_to_anchor=(0.0, 1.0), loc='upper left', fontsize=16)
    else:
        ax1.legend(bbox_to_anchor=(0.0, 0.0), loc='lower left', fontsize=16)
    fig.savefig(dir0+name0+".pdf")
    plt.close('all')

    return



def average_outputs(dir0, model, seeds, paths, modelname, num_classes, lowdiml, pltext, lowquant=0.25, highquant=0.75):
    print('investigate output')
    accuracy, detection, nparams, paramrs = [], [], [], []
    bool0 = np.array([True,
                      True, False,
                      True, False,
                      True, False, True, False,
                      True, False, True, False])
    for rnd, path0 in zip(seeds, paths):
        npz = np.load(dir0+path0+'output-scores.npz')
        accuracy.append(npz['accuracy'].T) # (len_rates, 2)
        detection.append(npz['detection']) # (len_rates, num_ood, 5)  # (auroc, aupr, aunr, fpr0.9, fpr0.95)

        ndims = npz['ndims'] # (len_rates)
        rates = npz['rates'] # (len_rates)

        dims = set_dims_model(modelname, num_classes)
        nparam, paramr = get_num_parameters(ndims, dims)
        nparams.append(nparam)
        paramrs.append(paramr)
    #
    accuracy = np.array(accuracy)
    detection = np.array(detection)
    nparams = np.array(nparams)
    paramrs = np.array(paramrs)

    accuracy1 = np.array([np.mean(accuracy, axis=0), np.std(accuracy, axis=0),
                          np.quantile(accuracy, 0.5, axis=0),
                          np.quantile(accuracy, lowquant, axis=0), np.quantile(accuracy, highquant, axis=0)])
    detection1 = np.array([np.mean(detection, axis=0), np.std(detection, axis=0),
                           np.quantile(detection, 0.5, axis=0),
                           np.quantile(detection, lowquant, axis=0), np.quantile(detection, highquant, axis=0)])
    nparams1 = np.array([np.mean(nparams, axis=0), np.std(nparams, axis=0),
                         np.quantile(nparams, 0.5, axis=0),
                         np.quantile(nparams, lowquant, axis=0), np.quantile(nparams, highquant, axis=0)])
    paramrs1 = np.array([np.mean(paramrs, axis=0), np.std(paramrs, axis=0),
                         np.quantile(paramrs, 0.5, axis=0),
                         np.quantile(paramrs, lowquant, axis=0), np.quantile(paramrs, highquant, axis=0)])

    tmp = detection1[:,0,:,bool0].transpose(1,2,0)
    
    np.set_printoptions(precision=5, suppress=True)
    center = tmp[2]
    width = np.abs(tmp[3:5] - np.array([tmp[2], tmp[2]]))
    width = (width[0] + width[1]) * 0.5
    # print(tmp[2])
    # print(tmp[3:5] - np.array([tmp[2], tmp[2]]))

    scores = np.array([[center, width]])
    # print(accuracy1[2:5])
    # print(nparams1[2:5])

    fig, ax1 = plt.subplots()
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('residual num. of params.', fontsize=20)
    ax1.set_ylabel('accuracy', fontsize=20)
    # ax1.set_xlim([1e-3, 1])
    # ax1.set_xscale('log')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.9, 1])
    for idx, (label0, color0) in enumerate(zip(['train', 'test'], ['black', 'red'])):
        tmp = accuracy1[:, idx, 1:]
        ax1.errorbar(nparams1[2], tmp[2],
                     xerr=np.abs(nparams1[3:5]-np.array([nparams1[2], nparams1[2]])),
                     yerr=np.abs(tmp[3:5]-np.array([tmp[2], tmp[2]])),
                     capsize=5, label=label0, linewidth=0.5, color=color0)
    ax1.legend()
    fig.savefig(dir0+f"accuracy.pdf")
    plt.close('all')

    fig, ax1 = plt.subplots()
    ax1.set_title(pltext+' '+model, fontsize=24)
    ax1.set_xlabel('threshold', fontsize=20)
    ax1.set_ylabel('residual num. of params.', fontsize=20)
    # ax1.set_xlim([1e-3, 1])
    ax1.set_xscale('log')
    ax1.set_xlim([1e-6, 1])
    ax1.set_ylim([0, 1])
    ax1.errorbar(rates, nparams1[2],
                 yerr=np.abs(nparams1[3:5]-np.array([nparams1[2], nparams1[2]])),
                 capsize=5, linewidth=0.5, color='black')
    # ax1.legend()
    fig.savefig(dir0+f"nparams.pdf")
    plt.close('all')



    return rates, accuracy1, nparams1, paramrs1, scores



def average_outputs_inference(dir0, seeds, epoch0, nc, traindata, datasets, colors, model, pltext, lowquant=0.25, highquant=0.75):
    # print('investigate output')
    np.set_printoptions(precision=5, suppress=True)
    datasets0, colors0 = [], []
    for dataset, color in zip(datasets, colors):
        if dataset=='traindata' or dataset==traindata:  continue
        datasets0.append(dataset)
        colors0.append(color)

    inference = []
    for dataset in datasets:
        infer = []
        # print(dataset)
        for rnd in seeds:
            logits = np.load(dir0+f'rnd{rnd}/outputs/logits{epoch0:0>4}_000_{dataset}.npz')['arr_0']
            # print(rnd, logits.shape)
            tmp = np.argmax(logits, axis=1)
            infer.append([np.count_nonzero(tmp==ic)/len(tmp) for ic in range(nc)])
        #
        infer = np.array(infer)
        coef_var = np.std(infer, axis=1) / np.mean(infer, axis=1)
        coef_qtl = (np.quantile(infer, highquant, axis=1) -  np.quantile(infer, lowquant, axis=1)) \
            / (np.quantile(infer, highquant, axis=1) +  np.quantile(infer, lowquant, axis=1))
        stats = [np.mean(infer, axis=0), np.std(infer, axis=0),
                np.quantile(infer, 0.5, axis=0),
                np.quantile(infer, lowquant, axis=0), np.quantile(infer, highquant, axis=0)]
        stat_cv = [np.mean(coef_var, axis=0), np.std(coef_var, axis=0),
                   np.quantile(coef_var, 0.5, axis=0),
                   np.quantile(coef_var, lowquant, axis=0), np.quantile(coef_var, highquant, axis=0)]
        stat_cq = [np.mean(coef_qtl, axis=0), np.std(coef_qtl, axis=0),
                   np.quantile(coef_qtl, 0.5, axis=0),
                   np.quantile(coef_qtl, lowquant, axis=0), np.quantile(coef_qtl, highquant, axis=0)]

        # print(dataset)
        # print(np.array(stats))
        inference.append(stats)

        center = [round(stat_cv[2], 3)] + [round(stat_cq[2], 3)] + [round(score, 3) for score in stats[2]]
        width = [round(0.5*(stat_cv[4] - stat_cv[3])*1000, 0)] + [round(0.5*(stat_cq[4] - stat_cq[3])*1000, 0)] \
            + [round(0.5*(score2-score1)*1000, 0) for score1, score2 in zip(stats[3], stats[4])]
        result = [f'{center0:.03f}({width0:.0f})' for center0, width0 in zip(center, width)]
        # print('   ', f'{dataset:<10}', result[0], result[1], result[2:])
        print('   ', f'{dataset:<10}', result[0], result[1])
    #
    inference = np.array(inference)

    datanums = [50000, 10000, 26032, 10000, 10000]

    center = np.array([infer[2] for infer, datanum in zip(inference, datanums)])
    width = np.array([[infer[2]-infer[3], infer[4]-infer[2]] for infer, datanum in zip(inference, datanums)])
    plt_inference(dir0, ['ID (train)', 'ID (test)'] + datasets0, np.arange(nc), center, width, colors, model, pltext, ymax=0.55)
    # print(inference)


    return





def average_mahalanobis(dir0, model, seeds, paths, datasets, traindata, layers,
                        dataref, color_dataset, pltext, lowdiml, print_result=False, lowquant=0.25, highquant=0.75):
    print('investigate_mahalanobis')
    name0 = ['tied']
    datasets0, colors0 = [], []
    for dataset, color in zip(datasets, color_dataset):
        if dataset=='traindata' or dataset==traindata:  continue
        datasets0.append(dataset)
        colors0.append(color)

    bool0 = np.array([True,
                      False, True,
                      False, True,
                      False, True, False, True,
                      False, True, False, True])
    diff0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    diff0 = np.array([diff0, diff0, diff0])
    np.set_printoptions(precision=5, suppress=True)

    results, results_best, results_last = [], [], []
    scores = []
    for iname in name0:
        results0, resultsdim0, results0_best, results0_last = [], [], [], []
        print(iname)
        for layer in layers:
            mahals, mahaldim, mahal_best, mahal_last = [], [], [], []
            lenmin = 1024
            idmaxes = []
            for irnd, path0 in zip(seeds, paths):
                npz = np.load(dir0+path0+'mahalanobis/mahalanobis-scores_'+iname+f'_{layer:0>2}.npz')
                mahal0 = npz['auroc']

                mahals.append(mahal0)
                mahaldim.append([mahal0, npz['dims']])

                len0 = len(npz['auroc'])
                mahal_last.append(mahal0[len0-1])

                idmax = np.argmin(mahal0[:,dataref,0])
                idmaxes.append(npz['dims'][idmax])
                mahal_best.append(mahal0[idmax])

                lenmin = min(lenmin, len(npz['dims']))

            mahals1 = np.array([mahal[:lenmin-1] for mahal in mahals])
            if layer==layers[-1]:  print(iname, layer, idmaxes, lenmin-1, mahaldim[-1][1][lenmin-2])
            
            mahalanobis_best = np.array([np.mean(mahal_best, axis=0), np.std(mahal_best, axis=0),
                                    np.quantile(mahal_best, 0.5, axis=0),
                                    np.quantile(mahal_best, lowquant, axis=0),
                                    np.quantile(mahal_best, highquant, axis=0)])
            mahalanobis_last = np.array([np.mean(mahal_last, axis=0), np.std(mahal_last, axis=0),
                                    np.quantile(mahal_last, 0.5, axis=0),
                                    np.quantile(mahal_last, lowquant, axis=0),
                                    np.quantile(mahal_last, highquant, axis=0)])
            mahalanobis = np.array([np.mean(mahals1, axis=0), np.std(mahals1, axis=0),
                                    np.quantile(mahals1, 0.5, axis=0),
                                    np.quantile(mahals1, lowquant, axis=0),
                                    np.quantile(mahals1, highquant, axis=0)])

            resultsdim0.append(mahaldim)
            results0.append(mahalanobis)
            results0_best.append(np.array(mahalanobis_best))
            results0_last.append(np.array(mahalanobis_last))
        results_best.append(results0_best)
        results_last.append(results0_last)
        results.append(results0)

        plt_dimension_mahalanobis(dir0, iname, paths, seeds, layers, resultsdim0,
                                  dataref, 'CIFAR', idx_score=0, scorename='AUROC')
        argmin0 = np.argmin(np.array(results0_best)[:,2,dataref,0])
        argmin1 = np.argmin(np.array(results0_last)[:,2,dataref,0])
        if print_result:
            tmp_best = np.array(results0_best)[argmin0, :, :, bool0].transpose(1,2,0)
            tmp_last = np.array(results0_last)[argmin1, :, :, bool0].transpose(1,2,0)
            argmins = np.argmin(results0[-1][2,:,dataref,0])
            tmp = results0[-1][:, argmins, :, bool0].transpose(1,2,0)
            print(tmp_best.shape, tmp_last.shape, tmp.shape)
            print('mahalanobis_best', argmin0)
            print(np.abs(diff0 - tmp_best[2]))
            print(tmp_best[3:5]-np.array([tmp_best[2], tmp_best[2]]))
            print('mahalanobis_last', argmin1)
            print(np.abs(diff0 - tmp_last[2]))
            print(tmp_last[3:5]-np.array([tmp_last[2], tmp_last[2]]))

            center = np.abs(diff0 - tmp[2])
            width = np.abs(tmp[3:5] - np.array([tmp[2], tmp[2]]))
            width = (width[0] + width[1]) * 0.5

            print('mahalanobis_stat', argmins, resultsdim0[-1][0][1][argmins])
            print(np.abs(diff0 - tmp[2]))
            print(tmp[3:5]-np.array([tmp[2], tmp[2]]))
            scores.append(np.array([center, width]))
    # results = np.array(results)
    results_best = np.array(results_best)
    results_last = np.array(results_last)
    print(results_best.shape, results_last.shape)


    name_score = 'AUROC'
    idx_score = 0
    results_best0 = results_best[:, :, :, :, idx_score]
    results_last0 = results_last[:, :, :, :, idx_score]


    center = results_best0[:, :, 2];  width = 0.5 * np.abs(results_best0[:, :, 3]-results_best0[:, :, 4])
    plt_mahalanobis_auc_layers(dir0, name_score, 'medi-best', name0,
                               datasets0, layers, center, width, colors0, lowdiml, model, pltext=pltext)
    center = results_last0[:, :, 2];  width = 0.5 * np.abs(results_last0[:, :, 3]-results_last0[:, :, 4])
    plt_mahalanobis_auc_layers(dir0, name_score, 'medi-last', name0,
                               datasets0, layers, center, width, colors0, lowdiml, model, pltext=pltext)

    center = results_best0[:, :, 0];  width = results_best0[:, :, 1]
    plt_mahalanobis_auc_layers(dir0, name_score, 'mean-best', name0,
                               datasets0, layers, center, width, colors0, lowdiml, model, pltext=pltext)
    center = results_last0[:, :, 0];  width = results_last0[:, :, 1]
    plt_mahalanobis_auc_layers(dir0, name_score, 'mean-last', name0,
                               datasets0, layers, center, width, colors0, lowdiml, model, pltext=pltext)

    return np.array(scores)


def average_projections(dir0, model, seeds, paths, nparams, paramrs, accuracy, layers,
                        datasets, traindata, colors, dataref, pltext, pltext1, lowdiml,
                        cm_mine, layers_custom, print_result=False,
                        lowquant=0.25, highquant=0.75):
    print('investigate_projection')
    datasets0, colors0 = [], []
    for dataset, color in zip(datasets, colors):
        if dataset=='traindata' or dataset==traindata:  continue
        datasets0.append(dataset)
        colors0.append(color)

    colors_cm = [cm_mine(layer/layers[-1]) for layer in layers]

    bool0 = np.array([True,
                      True, False,
                      True, False,
                      True, False, True, False,
                      True, False, True, False])

    np.set_printoptions(precision=5, suppress=True)
    scores = []
    for irnd, path0 in zip(seeds, paths):
        results = np.load(dir0+path0+'projection-scores.npz')['auc']
        scores.append(results)
    scores = np.array(scores)
    projections = np.array([np.mean(scores, axis=0), np.std(scores, axis=0),
                            np.quantile(scores, 0.5, axis=0),
                            np.quantile(scores, lowquant, axis=0),
                            np.quantile(scores, highquant, axis=0),
                            np.quantile(scores, 0.75, axis=0),
                            np.quantile(scores, 0.625, axis=0),
                            np.quantile(scores, 0.875, axis=0)])
    print('full-FC', projections[2:5, -1, 1, 1, :, bool0].transpose(1,2,0))

    idx_score = 0

    fig, ax1 = plt.subplots() # norm1 (all)
    ax1.set_title(pltext+r' $||x_p||$, '+model, fontsize=24)
    ax1.set_xlabel('residual num. of params.', fontsize=20)
    ax1.set_ylabel('AUROC', fontsize=20)
    # ax1.set_xlim([1e-3, 1])
    # ax1.set_xscale('log')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.5, 1])
    for idx, layer in enumerate(layers):
        tmp = projections[:, idx, 1::3, 0, dataref, idx_score]
        ax1.errorbar(nparams[2], tmp[2],
                     xerr=np.abs(nparams[3:5]-np.array([nparams[2], nparams[2]])),
                     yerr=np.abs(tmp[3:5]-np.array([tmp[2],tmp[2]])),
                     capsize=5, label=f'{layer:0>2}', color=colors_cm[idx], linewidth=0.5)
    custom_lines = [Line2D([0], [0], color=cm_mine(layer/layers_custom[-1]), lw=0.5) for layer in layers_custom]
    ax1.legend(custom_lines, layers_custom, bbox_to_anchor=(0.0, 1.0), loc='upper left', fontsize=16)
    # ax1.legend()
    fig.savefig(dir0+f"AUROC-norm1-all_CIFAR.pdf")
    plt.close('all')


    fig, ax1 = plt.subplots() # norm1 (all)
    ax1.set_title(pltext1+r' $||x_p|| / ||x||$, '+model, fontsize=24)
    ax1.set_xlabel('residual num. of params.', fontsize=20)
    ax1.set_ylabel('AUROC', fontsize=20)
    # ax1.set_xlim([1e-3, 1])
    # ax1.set_xscale('log')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.5, 1])
    for idx, layer in enumerate(layers):
        tmp = projections[:, idx, 1::3, 1, dataref, idx_score]
        ax1.errorbar(nparams[2], tmp[2],
                     xerr=np.abs(nparams[3:5]-np.array([nparams[2], nparams[2]])),
                     yerr=np.abs(tmp[3:5]-np.array([tmp[2],tmp[2]])),
                     capsize=5, label=f'{layer:0>2}', color=colors_cm[idx], linewidth=0.5)
    custom_lines = [Line2D([0], [0], color=cm_mine(layer/layers_custom[-1]), lw=0.5) for layer in layers_custom]
    ax1.legend(custom_lines, layers_custom, bbox_to_anchor=(0.0, 1.0), loc='upper left', fontsize=16)
    fig.savefig(dir0+f"AUROC-normr-all_CIFAR.pdf")
    plt.close('all')


    if print_result:
        for i in range(1, projections.shape[2], 3):
            print(i//3, projections[2, :, i, :, dataref, 0].T, paramrs[2, i//3])

    if traindata=='cifar10':
        dict_ilp = {'vgg13': 8, 'vgg16': 10, 'resnet18': 15, 'resnet34': 27}
        dict_ilr = {'vgg13': 10, 'vgg16': 13, 'resnet18': 17, 'resnet34': 33}
        threshp = {'vgg13': 0.85, 'vgg16': 0.85, 'resnet18': 0.85, 'resnet34': 0.85}
        threshr = {'vgg13': 0.85, 'vgg16': 0.85, 'resnet18': 0.85, 'resnet34': 0.85}
    elif traindata=='cifar100':
        dict_ilp = {'vgg13': 10, 'vgg16': 11, 'resnet18': 17, 'resnet34': 33}
        dict_ilr = {'vgg13': 10, 'vgg16': 13, 'resnet18': 17, 'resnet34': 33}
    elif traindata=='svhn':
        dict_ilp = {'vgg13': 8, 'vgg16': 10, 'resnet18': 15, 'resnet34': 29}
        dict_ilr = {'vgg13': 10, 'vgg16': 13, 'resnet18': 17, 'resnet34': 33}
        dict_icp = {'vgg13': 41, 'vgg16': 23, 'resnet18': 25, 'resnet34': 19}
        dict_icr = {'vgg13': 23, 'vgg16': 15, 'resnet18': 21, 'resnet34': 20}
        threshp = {'vgg13': 0.85, 'vgg16': 0.95, 'resnet18': 0.85, 'resnet34': 0.94}
        threshr = {'vgg13': 0.96, 'vgg16': 0.92, 'resnet18': 0.965, 'resnet34': 0.96}
    elif traindata=='mnist':
        dict_ilp = {'vgg13': 8, 'vgg16': 10, 'resnet18': 15, 'resnet34': 29}
        dict_ilr = {'vgg13': 10, 'vgg16': 13, 'resnet18': 17, 'resnet34': 33}


    ilayerp = dict_ilp[model];  ilayerr = dict_ilr[model]

    tmpp_full = projections[:, ilayerp, 1, 0, :, bool0].transpose(1,2,0)
    tmpr_full = projections[:, ilayerr, 1, 1, :, bool0].transpose(1,2,0)

    tmpp = projections[:, :, 1::3, 0]
    tmpr = projections[:, :, 1::3, 1]
    print(model, tmpp.shape)

    argmaxp = np.argmax(tmpp[2, ilayerp, :, dataref, idx_score])
    argmaxr = np.argmax(tmpr[2, ilayerr, :, dataref, idx_score])
    
    normp = tmpp[:, :, argmaxp, :, idx_score];  normr = tmpr[:, :, argmaxr, :, idx_score]

    print(paramrs.shape)
    paramrp = paramrs[:, argmaxp, ilayerp]; paramrr = paramrs[:, argmaxr, ilayerr]

    tmpp = tmpp[:, ilayerp, argmaxp, :, bool0].transpose(1,2,0)
    tmpr = tmpr[:, ilayerr, argmaxr, :, bool0].transpose(1,2,0)
    prmp = nparams[:, argmaxp];  prmr = nparams[:, argmaxr]
    accp = accuracy[:, 1, argmaxp];  accr = accuracy[:, 1, argmaxr]
    # tmp0p = nparams[:, argmaxp, ilayerp],  tmp0r = nparams[:, argmaxr, ilayerr]
    print(dir0)
    print(ilayerp, ilayerr, argmaxp, argmaxr)

    print(projections[2:5, ilayerp, 1, 0, :, bool0].transpose(1,2,0))

    print(np.array([prmp[2], prmp[3]-prmp[2], prmp[4]-prmp[2]]),
          np.array([paramrp[2], paramrp[3]-paramrp[2], paramrp[4]-paramrp[2]]),
          np.array([accp[2], accp[3]-accp[2], accp[4]-accp[2]]))
    print(tmpp[2])
    print(tmpp[3:5]-np.array([tmpp[2], tmpp[2]]))

    print(np.array([np.array(prmr[2]), prmr[3]-prmr[2], prmr[4]-prmr[2]]),
          np.array([paramrr[2], paramrr[3]-paramrr[2], paramrr[4]-paramrr[2]]),
          np.array([accr[2], accr[3]-accr[2], accr[4]-accr[2]]))
    print(tmpr[2])
    print(tmpr[3:5]-np.array([tmpr[2], tmpr[2]]))

    prmp0 = np.array([[prmp[2], prmp[3]-prmp[2], prmp[4]-prmp[2]],
                     [paramrp[2], paramrp[3]-paramrp[2], paramrp[4]-paramrp[2]]])
    prmr0 = np.array([[prmr[2], prmr[3]-prmr[2], prmr[4]-prmr[2]],
                     [paramrr[2], paramrr[3]-paramrr[2], paramrr[4]-paramrr[2]]])

    accp0 = np.array([accp[2], accp[3]-accp[2], accp[4]-accp[2]])
    accr0 = np.array([accr[2], accr[3]-accr[2], accr[4]-accr[2]])

    centerp = tmpp[2]
    widthp = np.abs(tmpp[3:5]-np.array([tmpp[2], tmpp[2]]))
    widthp = (widthp[0] + widthp[1]) * 0.5
    centerr = tmpr[2]
    widthr = np.abs(tmpr[3:5]-np.array([tmpr[2], tmpr[2]]))
    widthr = (widthr[0] + widthr[1]) * 0.5

    prms = np.array([prmp0, prmr0])
    accs = np.array([accp0, accr0])
    scores = np.array([[centerp, widthp], [centerr, widthr]])
    scores_full = np.array([[tmpp_full[2], 0.5*(tmpp_full[4]-tmpp_full[3])],
                            [tmpr_full[2], 0.5*(tmpr_full[4]-tmpr_full[3])]])

    center = [normp[2], normr[2]]
    width = [np.abs(np.array([normp[2],normp[2]]) - normp[3:5]),
             np.abs(np.array([normr[2],normr[2]]) - normr[3:5])]
    plt_projections_auc_layers(dir0, 'AUROC', 'medi',
                               [f'norm1-{ilayerp:0>2}', f'normr-{ilayerr:0>2}'], [pltext, pltext1],
                               datasets0, layers, center, width, colors0, lowdiml, model, diff0=0.0)
    print('projection-norm:', normp[2])
    print('projection-ratio:', normr[2])


    return prms, accs, scores, scores_full



def average_similarity(dir0, simname, scorename, model, seeds, paths, epoch0, layers, datasets, traindata, colors,
                       dataref, pltext, pltext1, lowdiml, vrange, vrange1, vrange2, vlabel0, idx_score=0, lowquant=0.25, highquant=0.75):
    print('investigate_stabilities: ')
    dataset0_idx = [idx for idx in range(len(datasets)) \
                    if datasets[idx]!='traindata' and datasets[idx]!=traindata]
    scorelabel = ['AUROC', 'AUPR', 'FPR90', 'FPR95']

    bool_reg0 = np.array([[(j <= i) * 1.0 for j in range(len(layers))] for i in range(len(layers))])
    bool_reg1 = np.array([[(j > i) * 1.0 for j in range(len(layers))] for i in range(len(layers))])
    bool_reg0 = np.array([bool_reg0] * len(datasets))
    bool_reg1 = np.array([bool_reg1] * len(datasets))
    print(bool_reg0.shape, bool_reg1.shape)
    # print(bool_reg0)
    # print(bool_reg1)

    datasets0 = [datasets[idx] for idx in dataset0_idx]
    colors0 = [colors[idx] for idx in dataset0_idx]

    data_plotted = datasets0[dataref]
    score_plotted = scorelabel[idx_score]

    scores, dims0, dims1, sranks, tranks = [], [], [], [], []
    for irnd, path0 in zip(seeds, paths):
        if 'reg' in simname:
            cka0 = np.load(dir0+path0+'similarity/reg0-all-norm_interlayers.npz')
            cka1 = np.load(dir0+path0+'similarity/reg1-all-norm_interlayers.npz')
        else:
            cka = np.load(dir0+path0+'similarity/'+simname+'-all-norm_interlayers.npz')

        if '-norm' in scorename:
            tmp = scorename.replace('-norm', '')
            scores.append(cka[tmp] / cka['dims'])
        else:
            if 'reg' in simname:
                scores.append(cka0[scorename]*bool_reg0 + cka1[scorename]*bool_reg1)
            else:
                scores.append(cka[scorename])

        dim0 = []
        for dataset in datasets:
            gram = np.load(dir0+path0+'features/gram-all-val{epoch0:0>4}-norm_'+dataset+'.npz')
            tmp0 = gram['dims'].reshape(-1)
            dim0.append(np.minimum.outer(tmp0, tmp0))
        dims0.append(np.array(dim0))

        if 'reg' in simname:
            dims1.append(cka0['dims']*bool_reg0 + cka1['dims']*bool_reg1)

            srank0 = cka0['frobenius']/cka0['spectrum']
            srank1 = cka1['frobenius']/cka1['spectrum']
            trank0 = cka0['trace']/cka0['spectrum']
            trank1 = cka1['trace']/cka1['spectrum']
            sranks.append(srank0*bool_reg0 + srank1*bool_reg1)
            tranks.append(trank0*bool_reg0 + trank1*bool_reg1)
        else:
            dims1.append(cka['dims'])
            sranks.append(cka['frobenius'] / cka['spectrum'])
            tranks.append(cka['trace'] / cka['spectrum'])

    scores, dims0, dims1 = np.array(scores), np.array(dims0), np.array(dims1)
    sranks, tranks = np.array(sranks), np.array(tranks)
    dims_ratio = dims1 / dims0

    similarity = np.array([np.mean(scores, axis=0), np.std(scores, axis=0),
                           np.quantile(scores, 0.5, axis=0),
                           np.quantile(scores, lowquant, axis=0),
                           np.quantile(scores, highquant, axis=0)])
    dims_gram = np.array([np.mean(dims0, axis=0), np.std(dims0, axis=0),
                          np.quantile(dims0, 0.5, axis=0),
                          np.quantile(dims0, lowquant, axis=0),
                          np.quantile(dims0, highquant, axis=0)])
    dims_sim = np.array([np.mean(dims1, axis=0), np.std(dims1, axis=0),
                         np.quantile(dims1, 0.5, axis=0),
                         np.quantile(dims1, lowquant, axis=0),
                         np.quantile(dims1, highquant, axis=0)])
    dims_ratio = np.array([np.mean(dims_ratio, axis=0), np.std(dims_ratio, axis=0),
                         np.quantile(dims_ratio, 0.5, axis=0),
                         np.quantile(dims_ratio, lowquant, axis=0),
                         np.quantile(dims_ratio, highquant, axis=0)])
    stablerank = np.array([np.mean(sranks*sranks, axis=0), np.std(sranks*sranks, axis=0),
                           np.quantile(sranks*sranks, 0.5, axis=0),
                           np.quantile(sranks*sranks, lowquant, axis=0),
                           np.quantile(sranks*sranks, highquant, axis=0)])
    tracerank = np.array([np.mean(tranks, axis=0), np.std(tranks, axis=0),
                          np.quantile(tranks, 0.5, axis=0),
                          np.quantile(tranks, lowquant, axis=0),
                          np.quantile(tranks, highquant, axis=0)])
    print(similarity.shape, dims_ratio.shape, np.amin(dims_ratio[2]), np.amax(dims_ratio[2]),
          np.amin(stablerank[2]), np.amax(stablerank[2]), np.amin(tracerank[2]), np.amax(tracerank[2]))



    nlayer = similarity.shape[-1]
    for idx, dataset in enumerate(datasets):
        if dataset==datasets0[dataref]:
            pltext0 = pltext1
        else:
            pltext0 = pltext

        if dataset=='traindata' or dataset==traindata:
            title0 = model+', ID'
        else:
            title0 = model+', OOD'

        center = similarity[2, idx]
        plt_similarity_interlayer(dir0, simname, 'medi', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange, [0.0, 0.9, 0.12, 0.9], vlabel0)

        center = dims_ratio[2, idx]
        # print(simname, model, dataset, center.shape, np.amin(center), np.amax(center))
        plt_similarity_interlayer(dir0, simname, 'medi-dims', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, [1e-2, 1e0], [0.0, 0.9, 0.12, 0.9], f'dim{vlabel0}/gram)')

        center = dims_gram[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-dims0', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, [1e-2, 1e0], [0.0, 0.9, 0.12, 0.9], f'dim(Gram)')

        center = dims_sim[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-dims1', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, [1e-2, 1e0], [0.0, 0.9, 0.12, 0.9], f'dim({vlabel0})')

        center = stablerank[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-srank', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange1, [0.0, 0.9, 0.12, 0.9], f'stable rank ({vlabel0})')

        center = tracerank[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-trank', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange2, [0.0, 0.9, 0.12, 0.9], f'trace rank ({vlabel0})')


    center = similarity[2,:,-1]
    width = np.abs(similarity[3:5,:,-1] - np.array([center,center]))
    plt_similarity_penultimate(dir0, simname, 'medi', layers, center, width,
                               ['ID (train)', 'ID (test)'] + datasets0, colors, pltext, lowdiml, model, vrange)

    return



def average_similarity0(dir0, simname, scorename, model, seeds, paths, layers, datasets, traindata, colors,
                        dataref, pltext, pltext1, lowdiml, vrange, vrange1, vrange2, vlabel0, idx_score=0, lowquant=0.25, highquant=0.75):
    print('investigate_stabilities: ')
    dataset0_idx = [idx for idx in range(len(datasets)) \
                    if datasets[idx]!='traindata' and datasets[idx]!=traindata]
    scorelabel = ['AUROC', 'AUPR', 'FPR90', 'FPR95']

    bool_reg0 = np.array([[(j <= i) * 1.0 for j in range(len(layers))] for i in range(len(layers))])
    bool_reg1 = np.array([[(j > i) * 1.0 for j in range(len(layers))] for i in range(len(layers))])
    bool_reg0 = np.array([bool_reg0] * len(datasets))
    bool_reg1 = np.array([bool_reg1] * len(datasets))
    print(bool_reg0.shape, bool_reg1.shape)
    # print(bool_reg0)
    # print(bool_reg1)

    datasets0 = [datasets[idx] for idx in dataset0_idx]
    colors0 = [colors[idx] for idx in dataset0_idx]

    data_plotted = datasets0[dataref]
    score_plotted = scorelabel[idx_score]

    scores, dims0, dims1, sranks, tranks = [], [], [], [], []
    for irnd, path0 in zip(seeds, paths):
        if 'reg' in simname:
            cka0 = np.load(dir0+path0+'similarity/reg0-all-norm_interlayers.npz')
            cka1 = np.load(dir0+path0+'similarity/reg1-all-norm_interlayers.npz')
        else:
            cka = np.load(dir0+path0+'similarity/'+simname+'-all-norm_interlayers.npz')

        if '-norm' in scorename:
            tmp = scorename.replace('-norm', '')
            scores.append(cka[tmp] / cka['dims'])
        else:
            if 'reg' in simname:
                scores.append(cka0[scorename]*bool_reg0 + cka1[scorename]*bool_reg1)
            else:
                scores.append(cka[scorename])

        if 'reg' in simname:
            dims1.append(cka0['dims']*bool_reg0 + cka1['dims']*bool_reg1)

            srank0 = cka0['frobenius']/cka0['spectrum']
            srank1 = cka1['frobenius']/cka1['spectrum']
            trank0 = cka0['trace']/cka0['spectrum']
            trank1 = cka1['trace']/cka1['spectrum']
            sranks.append(srank0*bool_reg0 + srank1*bool_reg1)
            tranks.append(trank0*bool_reg0 + trank1*bool_reg1)
        else:
            dims1.append(cka['dims'])
            sranks.append(cka['frobenius'] / cka['spectrum'])
            tranks.append(cka['trace'] / cka['spectrum'])

    scores, dims1 = np.array(scores), np.array(dims1)
    sranks, tranks = np.array(sranks), np.array(tranks)
    # dims_ratio = dims1 / dims0

    similarity = np.array([np.mean(scores, axis=0), np.std(scores, axis=0),
                           np.quantile(scores, 0.5, axis=0),
                           np.quantile(scores, lowquant, axis=0),
                           np.quantile(scores, highquant, axis=0)])
    # dims_gram = np.array([np.mean(dims0, axis=0), np.std(dims0, axis=0),
    #                       np.quantile(dims0, 0.5, axis=0),
    #                       np.quantile(dims0, lowquant, axis=0),
    #                       np.quantile(dims0, highquant, axis=0)])
    dims_sim = np.array([np.mean(dims1, axis=0), np.std(dims1, axis=0),
                         np.quantile(dims1, 0.5, axis=0),
                         np.quantile(dims1, lowquant, axis=0),
                         np.quantile(dims1, highquant, axis=0)])
    # dims_ratio = np.array([np.mean(dims_ratio, axis=0), np.std(dims_ratio, axis=0),
    #                      np.quantile(dims_ratio, 0.5, axis=0),
    #                      np.quantile(dims_ratio, lowquant, axis=0),
    #                      np.quantile(dims_ratio, highquant, axis=0)])
    stablerank = np.array([np.mean(sranks*sranks, axis=0), np.std(sranks*sranks, axis=0),
                           np.quantile(sranks*sranks, 0.5, axis=0),
                           np.quantile(sranks*sranks, lowquant, axis=0),
                           np.quantile(sranks*sranks, highquant, axis=0)])
    tracerank = np.array([np.mean(tranks, axis=0), np.std(tranks, axis=0),
                          np.quantile(tranks, 0.5, axis=0),
                          np.quantile(tranks, lowquant, axis=0),
                          np.quantile(tranks, highquant, axis=0)])
    # print(similarity.shape, dims_ratio.shape, np.amin(dims_ratio[2]), np.amax(dims_ratio[2]),
    #       np.amin(stablerank[2]), np.amax(stablerank[2]), np.amin(tracerank[2]), np.amax(tracerank[2]))



    nlayer = similarity.shape[-1]
    for idx, dataset in enumerate(datasets):
        if dataset==datasets0[dataref]:
            pltext0 = pltext1
        else:
            pltext0 = pltext

        if dataset=='traindata' or dataset==traindata:
            title0 = model+', ID'
        else:
            title0 = model+', OOD'

        center = similarity[2, idx]
        plt_similarity_interlayer(dir0, simname, 'medi', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange, [0.0, 0.9, 0.12, 0.9], vlabel0)

        center = dims_sim[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-dims1', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, [1e-2, 1e0], [0.0, 0.9, 0.12, 0.9], f'dim({vlabel0})')

        center = stablerank[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-srank', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange1, [0.0, 0.9, 0.12, 0.9], f'stable rank ({vlabel0})')

        center = tracerank[2, idx] * 1e-4
        plt_similarity_interlayer(dir0, simname, 'medi-trank', center, dataset, pltext0, [-nlayer*0.22, nlayer*0.95], title0, vrange2, [0.0, 0.9, 0.12, 0.9], f'trace rank ({vlabel0})')


    center = similarity[2,:,-1]
    width = np.abs(similarity[3:5,:,-1] - np.array([center,center]))
    plt_similarity_penultimate(dir0, simname, 'medi', layers, center, width,
                               ['ID (train)', 'ID (test)'] + datasets0, colors, pltext, lowdiml, model, vrange)

    return




def average_sensitivities(dir0, model, seeds, paths, datasets, traindata, layers0, colors, cm_mine,
                          layers_custom, dataref, pltext, pltext1, lowdiml,
                          idx_score=0, lowquant=0.25, highquant=0.75):
    print('investigate_stabilities: ')
    dataset0_idx = [idx for idx in range(len(datasets)) \
                    if datasets[idx]!='traindata' and datasets[idx]!=traindata]
    scorelabel = ['AUROC', 'AUPR', 'FPR90', 'FPR95']

    datasets0 = [datasets[idx] for idx in dataset0_idx]
    colors0 = [colors[idx] for idx in dataset0_idx]

    idx_test = datasets.index(traindata)

    # data_plotted = datasets0.index(data0)
    score_plotted = scorelabel[idx_score]

    # name1 = f'{idx-1:0>3}'
    detections = []
    statsl = []
    name1 = 'all'
    for layer0 in layers0:
        stats0 = np.array([np.load(dir0+path0+f'sensitivity/sensitivity{layer0:0>2}-scores_{name1}.npz')['stats']
                           for path0 in paths])  #auc, stats
        detections0 = np.array([np.load(dir0+path0+f'sensitivity/sensitivity{layer0:0>2}-scores_{name1}.npz')['auc']
                                for path0 in paths])  #auc, stats
        print(layer0, stats0.shape, detections0.shape)
        # stats0.shape = (rnd, output layer, input dataset, statistics)
        # detections0.shape = (rnd, output layer, ood dataset, detection score)

        statsl.append(np.array([np.mean(stats0, axis=0), np.std(stats0, axis=0),
                                np.quantile(stats0, 0.5, axis=0),
                                np.quantile(stats0, lowquant, axis=0),
                                np.quantile(stats0, highquant, axis=0)]))
        detections.append(np.array([np.mean(detections0, axis=0), np.std(detections0, axis=0),
                                    np.quantile(detections0, 0.5, axis=0),
                                    np.quantile(detections0, lowquant, axis=0),
                                    np.quantile(detections0, highquant, axis=0)]))
        # # [intersample stat, outlayer, intradataset stat, original dataset]

    xaxis = [layers0[idx_layer:]+[layers0[-1]+1] for idx_layer in range(len(layers0))]
    colors_cm = [cm_mine(layer0/layers0[-1]) for layer0 in layers0]

    # print([statl.shape for statl in statsl])
    for idx, dataset in enumerate(datasets):
        if dataset==datasets0[dataref]:
            pltext0 = pltext1
        else:
            pltext0 = pltext

        if dataset=='traindata' or dataset==traindata:
            title0 = model+', ID'
        else:
            title0 = model+', OOD'

        center = [statl[2, :, idx, 0] for statl in statsl]
        width = [np.abs(statl[3:5, :, idx, 0] - np.array([center0, center0]))
                 for statl, center0 in zip(statsl, center)]
        plt_sensitivity_stats(dir0, f'sensitivity-gauss_{dataset}_medi', layers0, xaxis, center, width,
                              colors_cm, layers_custom, cm_mine, lowdiml, pltext0, title0,
                              ylim=[1e-5,1e0], plt_legend=False)

    for idx, dataset in enumerate(datasets0):
        # detections0.shape = (rnd, output layer, ood dataset, detection score)
        center = [1.0 - detectl[2, :, idx, idx_score] for detectl in detections]
        width = [np.abs(1.0 - detectl[3:5, :, idx, idx_score] - np.array([center0, center0]))
                 for detectl, center0 in zip(detections, center)]
        plt_sensitivity_auc(dir0, f'sensitivity-gauss_{dataset}_{score_plotted}', layers0, xaxis, center, width,
                            colors_cm, layers_custom, cm_mine, lowdiml, pltext, model,
                            ylim=[0.5, 1.0], plt_legend=False)

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


def get_num_parameters(ndims, dims):
    dims1 = [3] + dims
    dims1[1] = 64
    len0 = len(dims1) - 1

    nparam0 = sum([9*dims1[0]*(9*dims1[0]+dims1[1])]\
                  + [dims1[i+1]*(9*dims1[i]+dims1[i+1]) for i in range(1,len0-1)] \
                  + [dims1[len0]*(dims1[len0-1]+dims1[len0])])

    ndims = ndims.T
    nparams = [sum([ndim[0] * (9*dims1[0]+dims1[1])]\
                   + [ndim[i] * (9*dims1[i]+dims1[i+1]) for i in range(1,len0-1)] \
                   + [ndim[len0-1]*(dims1[len0-1]+dims1[len0])]) / nparam0 for ndim in ndims]

    paramr = [[ndim[i] / dims[i] for i in range(len0)] for ndim in ndims]

    return np.array(nparams), np.array(paramr)


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
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in total')

    parser.add_argument('--rndmseed', default=0, type=int, help='index for random seed')
    parser.add_argument('--num_seed', default=10, type=int, help='index for random seed')

    return parser.parse_args(argv)


def print_parametercheck(args):
    print('=== {} ==='.format(args.model_name))
    print('traindata: {:s}'.format(args.traindata))
    print('total epochs: {:d}'.format(args.epochs))
    print('lr: {:f}'.format(args.lr))
    print('batch size: {:d}'.format(args.batchsize))

    return


def generate_cmap(colors):
    values = np.arange(len(colors)) / (len(colors)-1)
    clist = [(v, c) for v, c in zip(values, colors)]
    return LinearSegmentedColormap.from_list('custom_cmap', clist)




def main():
    args = parse_args(sys.argv[1:])
    print_parametercheck(args)
    dir0 = '/home/user/workdir/'
    epoch0 = args.epochs
    datasets = ['traindata', 'cifar10', 'svhn', 'mnist', 'cifar100']
    color_dataset = ['black', 'darkviolet', 'red', 'green', 'blue']
    colorbasis = ['salmon', 'indianred', 'darkred',
                  'red', 'darkorange', 'orange', 'yellow', 'greenyellow', 'limegreen',
                  'aquamarine', 'cyan', 'deepskyblue', 'dodgerblue', 'blue',
                  'darkblue', 'darkviolet', 'magenta', 'violet']
    cm_mine = generate_cmap(colorbasis)
    color_layer = None

    for dataset_in in ['cifar10', 'cifar100', 'svhn', 'mnist']:
        if dataset_in=='cifar100':
            num_classes = 100
        else:
            num_classes = 10

        if dataset_in=='cifar10':
            dict_lowdiml = {'vgg13': 7.5, 'vgg16': 9.5, 'resnet18': 15.5, 'resnet34': 27.5}
        elif dataset_in=='cifar100':
            dict_lowdiml = {'vgg13': 9.5, 'vgg16': 11.5, 'resnet18': 16.5, 'resnet34': 33.5}
        elif dataset_in=='svhn':
            dict_lowdiml = {'vgg13': 7.5, 'vgg16': 9.5, 'resnet18': 15.5, 'resnet34': 27.5}
        elif dataset_in=='mnist':
            dict_lowdiml = {'vgg13': 7.5, 'vgg16': 9.5, 'resnet18': 15.5, 'resnet34': 27.5}

        for modelname, pltext, pltext1 in zip(['vgg13', 'vgg16', 'resnet18', 'resnet34'],
                                              ['(a)', '(b)', '(c)', '(d)'],
                                              ['(e)', '(f)', '(g)', '(h)']):
            lowdiml = dict_lowdiml[modelname]

            dir1 = f'{dataset_in}/{modelname}/'
            if dataset_in=='cifar10' and modelname=='resnet34':
                seeds = [0] + [i for i in range(2, args.num_seed)]
            elif dataset_in=='svhn' and modelname=='vgg13':
                seeds = [0, 1, 2, 5, 7, 8]
            elif dataset_in=='svhn' and modelname=='vgg16':
                seeds = [0, 1, 3, 4, 5, 6, 7, 8]
            elif dataset_in=='mnist' and modelname=='resnet34':
                seeds = [0, 1] + [i for i in range(3, args.num_seed)]
            else:
                seeds = [i for i in range(args.num_seed)]
            paths = [f'rnd{i}/extract/' for i in seeds]

            weightname0, weightname = get_weight_dict_name(modelname)
            layers = [idx for idx in range(len(weightname))]
            layers0, layers_custom = set_layers_model(modelname)
            if dataset_in=='cifar10':
                dataref = 2
            elif dataset_in=='cifar100':
                dataref = 0
            else:
                dataref = 0


            print()
            print(dir0+dir1)
            average_outputs_inference(dir0+dir1, seeds, epoch0, num_classes, dataset_in,
                                      datasets, color_dataset, modelname, pltext)

            # continue

            rates, accuracy, nparams, paramrs, scores_prob \
                = average_outputs(dir0+dir1, modelname, seeds, paths, modelname, num_classes, lowdiml, pltext)
            print(rates.shape, accuracy.shape, nparams.shape, paramrs.shape, scores_prob.shape)
            acc_full = np.array([accuracy[2, 1, 0], 0.5*(accuracy[4, 1, 0]-accuracy[3, 1, 0])])

            # print(nparams[2], accuracy[2,1])
            average_covweight(dir0+dir1, modelname, seeds, paths, epoch0, layers, pltext, lowdiml)

            # [centering type,  layer,  stat type,  OOD dataset,  scores]
            scores_mahal \
                = average_mahalanobis(dir0+dir1, modelname, seeds, paths, datasets, dataset_in,
                                      layers, dataref, color_dataset, pltext, lowdiml, print_result=True)
            print(scores_mahal.shape)
            print()

            params_proj, acc_proj, scores_proj, scores_full \
                = average_projections(dir0+dir1, modelname, seeds, paths, nparams, paramrs, accuracy, layers,
                                      datasets, dataset_in, color_dataset, dataref,
                                      pltext, pltext1, lowdiml, cm_mine, layers_custom, print_result=False)
            print(params_proj.shape, scores_proj.shape)
            print()
            # continue

            idxdata0 = [idx for idx in range(len(datasets)) \
                        if datasets[idx]!='traindata' and datasets[idx]!=dataset_in]
            scorelabel = ['AUROC', 'AUPRi', 'AUPRo', 'FPR@TPR90', 'TPR@FPR10', 'FPR@TPR95', 'TPR@FPR05']
            datasets0 = [datasets[idx] for idx in idxdata0]

            scores = np.concatenate([scores_prob, scores_mahal, scores_proj])
            print(acc_full)
            print(scores_full)
            print(scores.shape)
            print(params_proj)
            print(acc_proj)

            for idata in range(3):
                print(datasets0[idata])
                for iscore in range(7):
                    tmp = scores[:, :, idata, iscore]
                    center = [round(score[0], 3) for score in tmp]
                    width = [round(score[1]*1000, 0) for score in tmp]
                    result = [f'{center0:.03f}({width0:.0f})' for center0, width0 in zip(center, width)]
                    print('   ', scorelabel[iscore], result)

            average_similarity0(dir0+dir1, 'cka', 'trace', modelname, seeds, paths, layers, datasets, dataset_in, color_dataset,
                                dataref, pltext, pltext1, lowdiml, [4e-2, 1e0], [1e-4, 5e-4], [1e-4, 1e-3], 'CKA', idx_score=0, lowquant=0.25, highquant=0.75)

            # average_similarity(dir0+dir1, 'cka', 'trace', modelname, seeds, paths, layers, datasets, dataset_in, color_dataset,
            #                    dataref, pltext, pltext1, lowdiml, [4e-2, 1e0], [1e-4, 5e-4], [1e-4, 1e-3], 'CKA', idx_score=0, lowquant=0.25, highquant=0.75)
            # average_similarity(dir0+dir1, 'cca', 'trace-norm', modelname, seeds, paths, layers, datasets, dataset_in, color_dataset,
            #                    dataref, pltext, pltext1, lowdiml, [2e-1, 1e0], [1e-2, 1e0], [1e-2, 1e0], 'CCA', idx_score=0, lowquant=0.25, highquant=0.75)
            # average_similarity(dir0+dir1, 'reg', 'trace', modelname, seeds, paths, layers, datasets, dataset_in, color_dataset,
            #                    dataref, pltext, pltext1, lowdiml, [4e-2, 1e0], [1e-4, 1e-3], [1e-4, 1e-2], 'LR', idx_score=0, lowquant=0.25, highquant=0.75)

            average_sensitivities(dir0+dir1, modelname, seeds, paths, datasets, dataset_in, layers0,
                                  color_dataset, cm_mine, layers_custom, dataref, pltext, pltext1, lowdiml,
                                  idx_score=0, lowquant=0.25, highquant=0.75)


            print()


    return


if __name__ == '__main__':
    main()
