import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import random
from typing import Optional, Callable, List

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import seaborn as sns
sns.set_theme()
sns.set_style('whitegrid')

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    print('pred: ', pred)
    
#     target_mod = target.topk(max(topk), 1, True, True)[1].t()
#     target_mod = target.view(1, -1)
#     print('target_mod: ', target_mod)
    expand = target.expand(-1, max(topk))
    print('expand: ', expand)
    
    correct = pred.eq(expand)
    print('correct: ', correct)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def sigmoid(x): 
    z = 1/(1 + np.exp(-x)) 
    return z

def compute_mcc(y_pred, y_true):
    """
    Compute MCC given y_pred and y_true. y_pred will be non-thresholded decision values.
    Need to convert to binary predictions by replacing top K values with 1, where
    K is the number of true positives.
    """
    # Make a copy first
    y_pred2 = np.copy(y_pred)
    y_true2 = np.copy(y_true)
    
    p = int(y_true2.sum())
    n = int(y_true2.shape[0] - p)
    
    thresh = np.sort(y_pred2)[-p]
    pos_mask = np.where((y_pred2 >= thresh))[0]
    neg_mask = np.where((y_pred2 < thresh))[0]
    y_pred2[pos_mask] = 1.
    y_pred2[neg_mask] = -1
    
    mask = np.where((y_true2 == 0.))[0]
    y_true2[mask] = -1.
    
    return matthews_corrcoef(y_true2, y_pred2)
    

''' ROC CURVE '''
def plot_roc(y_pred, y_true, roc_name, plot=False, rad_xs=None, rad_ys=None):
    # given the test_ground_truth, and test_predictions 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)

    if plot: 
        plt.figure(dpi=100)
        plt.title(roc_name)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.gca().set_aspect('equal', adjustable='box')
        
        if rad_xs is not None and rad_ys is not None:
            plt.scatter(rad_xs, rad_ys, c=['c', 'm', 'y'])
            
        plt.show()
        # plt.savefig(f'DRACON_{roc_name}.png', dpi=1000, bbox_inches='tight')
        
    return fpr, tpr, thresholds, roc_auc

# J = TP/(TP+FN) + TN/(TN+FP) - 1 = tpr - fpr
def choose_operating_point(fpr, tpr, thresholds):
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1-_fpr
            J = _tpr - _fpr
#     print(J)
    return sens, spec

''' PRECISION-RECALL CURVE '''
def plot_pr(y_pred, y_true, pr_name, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true==1]) / len(y_true)
    
    if plot: 
        plt.figure(dpi=80)
        plt.title(pr_name)
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % pr_auc)
        # axis labels
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [baseline, baseline],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the plot
        plt.show()
    return precision, recall, thresholds

def evaluate(y_pred, y_true, cxr_labels,
             plot=False,
             roc_name='Receiver Operating Characteristic',
             pr_name='Precision-Recall Curve'):
    
    '''
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)
    
    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1. 
    
    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr. 
    
    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity, 
    PR-Curve, Precision, Recall for each class. 
    '''
    import warnings
    warnings.filterwarnings('ignore')

#     print(y_pred.shape, y_true.shape)
    assert(y_pred.shape == y_true.shape)
    num_classes = y_pred.shape[-1]
    
    auc_dataframes = []
    mcc_dataframes = []
    
    radiologists = pd.read_csv('radiologists.csv')
    
    for i in range(num_classes): 
#         print('{}.'.format(cxr_labels[i]))
        y_pred_i = y_pred[:, i] # (num_samples,)
        y_true_i = y_true[:, i] # (num_samples,)
        
        cxr_label = cxr_labels[i]
        
        rad = radiologists[cxr_label]
        xs = rad[:3]
        ys = rad[3:]
        
        ''' MCC '''
        mcc = compute_mcc(y_pred_i, y_true_i)
        mcc_results = [[mcc]]
        mcc_df = pd.DataFrame(mcc_results, columns=[cxr_label+'_mcc'])
        mcc_dataframes.append(mcc_df)
        
        ''' ROC CURVE '''
        # roc_name = cxr_labels[i] + ' ROC Curve'
        roc_name = cxr_labels[i]
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot=plot, rad_xs=xs, rad_ys=ys)
        
#         print('AUROC: '.format(cxr_labels[i]) + str(roc_auc))
        sens, spec = choose_operating_point(fpr, tpr, thresholds)
#         print('Sensitivity: ' + str(sens))
#         print('Specificity: ' + str(spec))

        auc_results = [[roc_auc]]
        auc_df = pd.DataFrame(auc_results, columns=[cxr_label+'_auc'])
        auc_dataframes.append(auc_df)
        
        
        
        ''' PRECISION-RECALL CURVE '''
        pr_name = cxr_labels[i] + ' Precision-Recall Curve'
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name, plot=plot)
        
#         print('\n')
    #     p = compute_precision(test_y, test_pred)
    #     print('Average Precision: ' + str(p))
    auc_dfs = pd.concat(auc_dataframes, axis=1)
    mcc_dfs = pd.concat(mcc_dataframes, axis=1)
    return auc_dfs, mcc_dfs

''' Bootstrap and Confidence Intervals '''
def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df
    
def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000, confidence_level=0.05): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    '''
    y_pred # (500, 14)
    y_true # (500, 14)
    
    idx = np.arange(len(y_true))
    
    boot_auc_stats = []
    boot_mcc_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        auc_stats, mcc_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels)
        auc_stats.drop(columns=['No Finding_auc'], inplace=True)
        auc_stats['Mean_auc'] = auc_stats.mean(axis=1)
        mcc_stats.drop(columns=['No Finding_mcc'], inplace=True)
        mcc_stats['Mean_mcc'] = mcc_stats.mean(axis=1)
        
        boot_auc_stats.append(auc_stats)
        boot_mcc_stats.append(mcc_stats)

    boot_auc_stats = pd.concat(boot_auc_stats) # pandas array of evaluations for each sample
    boot_mcc_stats = pd.concat(boot_mcc_stats) # pandas array of evaluations for each sample
    return boot_auc_stats, boot_mcc_stats, compute_cis(boot_auc_stats, confidence_level=confidence_level), compute_cis(boot_mcc_stats, confidence_level=confidence_level)


def bootstrap_avg(y_preds, y_true, cxr_labels, n_samples=1000, confidence_level=0.05, avg_fn='mean'): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    '''
    # y_preds # (10, 500, 14)
    # y_true # (500, 14)
    
    idx = np.arange(len(y_true))
    
    boot_auc_stats = []
    boot_mcc_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True)
        y_true_sample = y_true[sample]
        auc_stats_list = []
        mcc_stats_list = []
        # print("LEN Y PREDS", len(y_preds), y_preds.shape)
        for j in range(len(y_preds)):  # 10 in this case
            y_pred_sample = y_preds[j][sample]

            auc_stats, mcc_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels)
            auc_stats.drop(columns=['No Finding_auc'], inplace=True)
            auc_stats['Mean_auc'] = auc_stats.mean(axis=1)
            mcc_stats.drop(columns=['No Finding_mcc'], inplace=True)
            mcc_stats['Mean_mcc'] = mcc_stats.mean(axis=1)
            
            # print("AUC STATS1", auc_stats.shape)
            # print(auc_stats)
            auc_stats_list.append(auc_stats)
            mcc_stats_list.append(mcc_stats)
        
        # Take the average of the models' outputs
        auc_stats = pd.concat(auc_stats_list, axis=0)
        mcc_stats = pd.concat(mcc_stats_list, axis=0)
        
        # print("AUC STATS2", auc_stats.shape)
        # print(auc_stats)
        
        if avg_fn == 'mean':
            auc_stats = auc_stats.mean(axis=0).to_frame().T
            mcc_stats = mcc_stats.mean(axis=0).to_frame().T
        elif avg_fn == 'median':
            auc_stats = auc_stats.median(axis=0).to_frame().T
            mcc_stats = mcc_stats.median(axis=0).to_frame().T
        
        # print("AUC STATS3", auc_stats.shape)
        # print(auc_stats)
            
            
        boot_auc_stats.append(auc_stats)
        boot_mcc_stats.append(mcc_stats)
    
    boot_auc_stats = pd.concat(boot_auc_stats, axis=0) # pandas array of evaluations for each sample
    boot_mcc_stats = pd.concat(boot_mcc_stats, axis=0) # pandas array of evaluations for each sample
    # print("BOOT AUC STATS", boot_auc_stats.shape)
    # print(boot_auc_stats)

    return boot_auc_stats, boot_mcc_stats, compute_cis(boot_auc_stats, confidence_level=confidence_level), compute_cis(boot_mcc_stats, confidence_level=confidence_level)



def bootstrap_comparison(y_pred1, y_pred2, y_true, cxr_labels, n_samples=1000, confidence_level=0.05): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    '''
    
    idx = np.arange(len(y_true))
    
    boot_auc_stats = []
    boot_mcc_stats = []
    for i in tqdm(range(n_samples)):
        sample = resample(idx, replace=True)
        y_pred1_sample = y_pred1[sample]
        y_pred2_sample = y_pred2[sample]
        y_true_sample = y_true[sample]
        
        auc_stats1, mcc_stats1 = evaluate(y_pred1_sample, y_true_sample, cxr_labels)
        auc_stats1.drop(columns=['No Finding_auc'], inplace=True)
        auc_stats1['Mean_auc'] = auc_stats1.mean(axis=1)
        mcc_stats1.drop(columns=['No Finding_mcc'], inplace=True)
        mcc_stats1['Mean_mcc'] = mcc_stats1.mean(axis=1)
        
        
        auc_stats2, mcc_stats2 = evaluate(y_pred2_sample, y_true_sample, cxr_labels)
        auc_stats2.drop(columns=['No Finding_auc'], inplace=True)
        auc_stats2['Mean_auc'] = auc_stats2.mean(axis=1)
        mcc_stats2.drop(columns=['No Finding_mcc'], inplace=True)
        mcc_stats2['Mean_mcc'] = mcc_stats2.mean(axis=1)
        
        auc_stats = auc_stats1 - auc_stats2 # difference between two models
        mcc_stats = mcc_stats1 - mcc_stats2
        
        boot_auc_stats.append(auc_stats)
        boot_mcc_stats.append(mcc_stats)

    boot_auc_stats = pd.concat(boot_auc_stats) # pandas array of evaluations for each sample
    boot_mcc_stats = pd.concat(boot_mcc_stats)
    return boot_auc_stats, boot_mcc_stats, compute_cis(boot_auc_stats, confidence_level=confidence_level), compute_cis(boot_mcc_stats, confidence_level=confidence_level)

    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
class CXRTestDataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        cxr_labels: List of possible labels (there should be 14 for CheXpert).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, label_path, cxr_labels, transform=None, cutlabels=True):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        full_labels = pd.read_csv(label_path)
        print("FULL LABELS", full_labels.columns)
        if cutlabels: 
            full_labels = full_labels.loc[:, cxr_labels]
        else: 
            full_labels.drop(full_labels.columns[0], axis=1, inplace=True)
        self.labels = full_labels.to_numpy()
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.from_numpy(self.labels[idx])
        sample = {'img': img, 'label': label }
    
        return sample

    

class TorchDataset(Dataset):
    """For converting from h5 to PyTorch datasets"""
    def __init__(self, dset: CXRTestDataset, indices = None):
        if indices is None:
            indices = range(len(dset))
        
        self.dset = dset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.dset[self.indices[idx]]['img']
        label = self.dset[self.indices[idx]]['label']
        
        label = torch.nan_to_num(label, nan=0.)  # replace blank with 0
        label = torch.abs(label)  # replace -1 (uncertain) with 1 (present)
        sample = (image, label)
        return sample

    
def load_chexpert(cxr_labels, batch_size, pretrained=True, train_percent: Optional[float] = 0.01, K: Optional[int] = None):
    if pretrained:
        input_resolution = 224  # for pretrained model
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
    else:
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])

    chexpert_test = CXRTestDataset('/deep/group/data/med-data/test_cxr.h5',
                                   '/deep/group/data/med-data/final_paths.csv',
                                    cxr_labels,
                                    transform,
                                  )
    chexpert_train = CXRTestDataset('/deep/group/img-graph/CheXpert/train.h5',
                                    '/deep/group/data/med-data/train.csv',
                                    cxr_labels,
                                    transform,
                                   )
    chexpert_val = CXRTestDataset('/deep/group/img-graph/CheXpert/valid.h5',
                                  '/deep/group/data/CheXpert-320x320/valid.csv',
                                  cxr_labels,
                                  transform,
                                 )
    
    indices = range(len(chexpert_train))
    
    # few-shot learning
    print("K", K, "train_percent", train_percent)
    if K is not None:
        condition_count = 0  # count the number of images for a given condition found so far
        train_indices = []
        N = len(chexpert_train)

        for j in range(len(cxr_labels)):
            condition_count = 0
            while condition_count < K:
                i = random.randint(0, N-1)
                if i in train_indices:  # don't reuse indices
                    continue
                
                # This means condition is present
                if chexpert_train[i]['label'][j].item() == 1:
                    train_indices.append(i)
                    condition_count += 1
    else:
        train_indices, _ = train_test_split(indices, test_size=1-train_percent, random_state=42)
    
    train_dset = TorchDataset(chexpert_train, train_indices)
    val_dset = TorchDataset(chexpert_val)
    test_dset = TorchDataset(chexpert_test)
    
    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader


def downstream_final_eval(dataloader, model, loss_fn, cxr_labels, device, n_samples, confidence_level):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    y_pred = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            
#             print("PREDICTION")
#             print(pred)
#             print("y!!!!")
#             print(y)
            
            test_loss += loss_fn(pred, y).item()
            y_pred.extend(pred.to('cpu').numpy())
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss /= size
    
    y_pred, y_actual = np.array(y_pred), np.array(y_actual)
    auc_df, mcc_df = evaluate(y_pred, y_actual, cxr_labels, plot=True)
    auc_df, mcc_df = auc_df.astype(float), mcc_df.astype(float)
    auc_df.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df)
    mcc_df.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df)
    mean_auroc = auc_df.mean(axis=1)[0]
    mean_mcc = mcc_df.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc:>4f}, Mean MCC: {mean_mcc:>4f}, Avg loss: {test_loss:>8f} \n")
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap(y_pred, y_actual,
                                                                             cxr_labels, n_samples=n_samples,
                                                                             confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc, mean_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals


def downstream_ensemble_final_eval(dataloader, models, loss_fn, cxr_labels, device, ensemble_fn,
                                   n_samples, confidence_level):
    for model in models:
        model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    y_pred = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            preds = []
            for model in models:
                model.to(device)
                model_pred = model(X)
                preds.append(model_pred.to('cpu').numpy())
                model.to(torch.device('cpu'))  # free up space on CUDA
            preds = np.array(preds)
            if ensemble_fn == 'median':
                ensemble_pred = np.median(preds, axis=0)
            elif ensemble_fn == 'mean':
                ensemble_pred = np.mean(preds, axis=0)
            elif ensemble_fn == 'min':
                ensemble_pred = np.min(preds, axis=0)
            elif ensemble_fn == 'max':
                ensemble_pred = np.max(preds, axis=0)
            
            print("ENSEMBLE PRED DIM", ensemble_pred.shape)
            test_loss += loss_fn(torch.tensor(ensemble_pred).to(device), y).item()
            y_pred.extend(ensemble_pred)
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss /= size
    
    y_pred, y_actual = np.array(y_pred), np.array(y_actual)
    auc_df, mcc_df = evaluate(y_pred, y_actual, cxr_labels, plot=True)
    auc_df, mcc_df = auc_df.astype(float), mcc_df.astype(float)
    auc_df.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df)
    mcc_df.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df)
    mean_auroc = auc_df.mean(axis=1)[0]
    mean_mcc = mcc_df.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc:>4f}, Mean MCC: {mean_mcc:>4f}, Avg loss: {test_loss:>8f} \n")
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap(y_pred, y_actual, cxr_labels, n_samples=n_samples, confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc, mean_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals


def downstream_concat_final_eval(dataloader, models, loss_fn, cxr_labels, device,
                                   n_samples, confidence_level):
    for model in models:
        model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    y_pred = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            
            for model in models:
                model.to(device)
                pred = model(X)
                y_pred.extend(pred.to('cpu').numpy())
                model.to(torch.device('cpu'))  # free up space on CUDA
                test_loss += loss_fn(pred, y).item()
                
                y_actual.extend(y.to('cpu').numpy())
    
    test_loss /= size * len(models)
    
    y_pred, y_actual = np.array(y_pred), np.array(y_actual)
    auc_df, mcc_df = evaluate(y_pred, y_actual, cxr_labels, plot=True)
    auc_df, mcc_df = auc_df.astype(float), mcc_df.astype(float)
    auc_df.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df)
    mcc_df.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df)
    mean_auroc = auc_df.mean(axis=1)[0]
    mean_mcc = mcc_df.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc:>4f}, Mean MCC: {mean_mcc:>4f}, Avg loss: {test_loss:>8f} \n")
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap(y_pred, y_actual, cxr_labels, n_samples=n_samples, confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc, mean_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals


def downstream_avg_final_eval(dataloader, models, loss_fn, cxr_labels, device,
                               n_samples, confidence_level):
    y_preds = []
    for model in models:
        model.eval()
        y_preds.append([])
    size = len(dataloader.dataset)
    test_losses = np.zeros(len(models))
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            
            for i in range(len(models)):
                models[i].to(device)
                pred = models[i](X)
                y_preds[i].extend(pred.to('cpu').numpy())
                models[i].to(torch.device('cpu'))  # free up space on CUDA
                test_losses[i] += loss_fn(pred, y).item()
            y_actual.extend(y.to('cpu').numpy())
    
    test_losses /= size
    
    y_preds, y_actual = np.array(y_preds), np.array(y_actual)
    # Don't need to show the AUROC curve for all the models
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap_avg(y_preds, y_actual, cxr_labels, n_samples=n_samples, confidence_level=confidence_level)
    # print(boot_stats)

    return boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals



def downstream_ensemble_comparison_final_eval(dataloader, models1, models2, loss_fn, cxr_labels,
                                              device, ensemble_fn, n_samples, confidence_level):
    for model in models1:
        model.eval()
    for model in models2:
        model.eval()
        
    size = len(dataloader.dataset)
    test_loss1 = 0
    test_loss2 = 0
    y_pred1 = []
    y_pred2 = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            preds1 = []
            preds2 = []
            for model in models1:
                model.to(device)
                model_pred1 = model(X)
                preds1.append(model_pred1.to('cpu').numpy())
                model.to(torch.device('cpu'))  # free up space on CUDA
            for model in models2:
                model.to(device)
                model_pred2 = model(X)
                preds2.append(model_pred2.to('cpu').numpy())
                model.to(torch.device('cpu'))
            
            preds1 = np.array(preds1)
            preds2 = np.array(preds2)
            if ensemble_fn == 'median':
                ensemble_pred1 = np.median(preds1, axis=0)
                ensemble_pred2 = np.median(preds2, axis=0)
            elif ensemble_fn == 'mean':
                ensemble_pred1 = np.mean(preds1, axis=0)
                ensemble_pred2 = np.mean(preds1, axis=0)
            elif ensemble_fn == 'min':
                ensemble_pred1 = np.min(preds1, axis=0)
                ensemble_pred2 = np.min(preds2, axis=0)
            elif ensemble_fn == 'max':
                ensemble_pred1 = np.max(preds1, axis=0)
                ensemble_pred2 = np.max(preds2, axis=0)
            
            # print("ENSEMBLE PRED DIM", ensemble_pred1.shape)
            test_loss1 += loss_fn(torch.tensor(ensemble_pred1).to(device), y).item()
            test_loss2 += loss_fn(torch.tensor(ensemble_pred2).to(device), y).item()
            y_pred1.extend(ensemble_pred1)
            y_pred2.extend(ensemble_pred2)
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss1 /= size
    test_loss2 /= size
    
    y_pred1, y_pred2, y_actual = np.array(y_pred1), np.array(y_pred2), np.array(y_actual)
    auc_df1, mcc_df1 = evaluate(y_pred1, y_actual, cxr_labels, plot=True)
    auc_df1, mcc_df1 = auc_df1.astype(float), mcc_df1.astype(float)
    auc_df1.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df1)
    mcc_df1.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df1)
    mean_auroc1 = auc_df1.mean(axis=1)[0]
    mean_mcc1 = mcc_df1.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc1:>4f}, Mean MCC: {mean_mcc1:>4f}, Avg loss: {test_loss1:>8f} \n")
    
    auc_df2, mcc_df2 = evaluate(y_pred2, y_actual, cxr_labels, plot=True)
    auc_df2, mcc_df2 = auc_df2.astype(float), mcc_df2.astype(float)
    auc_df2.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df2)
    mcc_df2.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df2)
    mean_auroc2 = auc_df2.mean(axis=1)[0]
    mean_mcc2 = mcc_df2.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc2:>4f}, Mean MCC: {mean_mcc2:>4f}, Avg loss: {test_loss2:>8f} \n")
    
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap_comparison(y_pred1, y_pred2, y_actual,
                                                                                        cxr_labels, n_samples=n_samples,
                                                                                        confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc1, mean_auroc2, mean_mcc1, mean_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals


def downstream_comparison_final_eval(dataloader, model1, model2, loss_fn, cxr_labels, device,
                                     n_samples, confidence_level):
    """
    Compare any two models
    """
    model1.eval()
    model2.eval()
    size = len(dataloader.dataset)
    test_loss1 = 0
    test_loss2 = 0
    y_pred1 = []
    y_pred2 = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred1 = model1(X)
            pred2 = model2(X)
            
            test_loss1 += loss_fn(pred1, y).item()
            test_loss2 += loss_fn(pred2, y).item()
            y_pred1.extend(pred1.to('cpu').numpy())
            y_pred2.extend(pred2.to('cpu').numpy())
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss1 /= size
    test_loss2 /= size
    
    y_pred1, y_pred2, y_actual = np.array(y_pred1), np.array(y_pred2), np.array(y_actual)
    auc_df1, mcc_df1 = evaluate(y_pred1, y_actual, cxr_labels, plot=True)
    auc_df1, mcc_df1 = auc_df1.astype(float), mcc_df1.astype(float)
    auc_df1.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df1)
    mcc_df1.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df1)
    mean_auroc1 = auc_df1.mean(axis=1)[0]
    mean_mcc1 = mcc_df1.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc1:>4f}, Mean MCC: {mean_mcc1:>4f}, Avg loss: {test_loss1:>8f} \n")
    
    auc_df2, mcc_df2 = evaluate(y_pred2, y_actual, cxr_labels, plot=True)
    auc_df2, mcc_df2 = auc_df2.astype(float), mcc_df2.astype(float)
    auc_df2.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df2)
    mcc_df2.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df2)
    mean_auroc2 = auc_df2.mean(axis=1)[0]
    mean_mcc2 = mcc_df2.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc2:>4f}, Mean MCC: {mean_mcc2:>4f}, Avg loss: {test_loss2:>8f} \n")
    
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap_comparison(y_pred1, y_pred2, y_actual,
                                                                                        cxr_labels, n_samples=n_samples,
                                                                                        confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc1, mean_auroc2, mean_mcc1, mean_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals



def downstream_concat_comparison_final_eval(dataloader, models1, models2, loss_fn, cxr_labels,
                                              device, n_samples, confidence_level):
    for model in models1:
        model.eval()
    for model in models2:
        model.eval()
        
    size = len(dataloader.dataset)
    test_loss1 = 0
    test_loss2 = 0
    y_pred1 = []
    y_pred2 = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            preds1 = []
            preds2 = []
            for model in models1:
                model.to(device)
                pred1 = model(X)
                y_pred1.extend(pred1.to('cpu').numpy())
                model.to(torch.device('cpu'))  # free up space on CUDA
                test_loss1 += loss_fn(pred1, y).item()
                
                y_actual.extend(y.to('cpu').numpy())
                
            for model in models2:
                model.to(device)
                pred2 = model(X)
                y_pred2.extend(pred2.to('cpu').numpy())
                model.to(torch.device('cpu'))  # free up space on CUDA
                test_loss2 += loss_fn(pred2, y).item()

    
    test_loss1 /= size * len(models1)
    test_loss2 /= size * len(models2)
    
    y_pred1, y_pred2, y_actual = np.array(y_pred1), np.array(y_pred2), np.array(y_actual)
    auc_df1, mcc_df1 = evaluate(y_pred1, y_actual, cxr_labels, plot=True)
    auc_df1, mcc_df1 = auc_df1.astype(float), mcc_df1.astype(float)
    auc_df1.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df1)
    mcc_df1.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df1)
    mean_auroc1 = auc_df1.mean(axis=1)[0]
    mean_mcc1 = mcc_df1.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc1:>4f}, Mean MCC: {mean_mcc1:>4f}, Avg loss: {test_loss1:>8f} \n")
    
    auc_df2, mcc_df2 = evaluate(y_pred2, y_actual, cxr_labels, plot=True)
    auc_df2, mcc_df2 = auc_df2.astype(float), mcc_df2.astype(float)
    auc_df2.drop(columns=['No Finding_auc'], inplace=True)
    print(auc_df2)
    mcc_df2.drop(columns=['No Finding_mcc'], inplace=True)
    print(mcc_df2)
    mean_auroc2 = auc_df2.mean(axis=1)[0]
    mean_mcc2 = mcc_df2.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc2:>4f}, Mean MCC: {mean_mcc2:>4f}, Avg loss: {test_loss2:>8f} \n")
    
    
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = bootstrap_comparison(y_pred1, y_pred2, y_actual,
                                                                                        cxr_labels, n_samples=n_samples,
                                                                                        confidence_level=confidence_level)
    # print(boot_stats)
    print(auc_intervals)
    print(mcc_intervals)
    return mean_auroc1, mean_auroc2, mean_mcc1, mean_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals



def chexpert_eval(model, config, device, downstream_model_path, cxr_labels=None, n_samples=1000, confidence_level=0.05):
    """
    model: model loaded
    config: dictionary containing
        'batch_size', 'loss_fn', 'optimizer', 'train_percent', 'epochs', and 'momentum' (if SGD)
    device: Either torch.device('cuda:0') or torch.device('cpu')
    downstream_model_path: path to save best performing downstream model
    """
    # cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other',
    #               'Pneumonia', 'Pneumothorax', 'Support Devices']
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema','No Finding','Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )

    downstream_model = model.to(device)
    # Reload downstream model with best parameters according to validation AUROC
    downstream_model.load_state_dict(torch.load(downstream_model_path))
    test_auroc, test_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_final_eval(test_dataloader,
                                                                                                     downstream_model,
                                                                                                     loss_fn, cxr_labels,
                                                                                                     device,
                                                                                                            n_samples=1000, confidence_level=0.05
                                                                                                               
                                                                                                              )
    return test_auroc, test_mcc, downstream_model, auc_intervals, mcc_intervals


def chexpert_ensemble_eval(models, config, device, downstream_model_paths, cxr_labels=None, ensemble_fn='median', n_samples=1000, confidence_level=0.05):
    """
    Like chexpert_eval but ensembles the models
    """
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema','No Finding','Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT ENSEMBLE EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )
    
    # Reload downstream model with best parameters according to validation AUROC
    downstream_models = []
    for i in range(len(models)):
        downstream_model = models[i]
        downstream_model.load_state_dict(torch.load(downstream_model_paths[i]))
        downstream_models.append(downstream_model)
        
    test_auroc, test_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_ensemble_final_eval(test_dataloader, downstream_models,
                                                                 loss_fn, cxr_labels, device, ensemble_fn,
                                                                                                                        n_samples=n_samples, confidence_level=confidence_level)
    return test_auroc, test_mcc, downstream_model, auc_intervals, mcc_intervals


def chexpert_concat_eval(models, config, device, downstream_model_paths, cxr_labels=None,
                         n_samples=1000, confidence_level=0.05):
    """
    Like chexpert_eval but concatenates the model predictions
    """
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema','No Finding','Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT CONCAT EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )
    
    # Reload downstream model with best parameters according to validation AUROC
    downstream_models = []
    for i in range(len(models)):
        downstream_model = models[i]
        downstream_model.load_state_dict(torch.load(downstream_model_paths[i]))
        downstream_models.append(downstream_model)
        
    test_auroc, test_mcc, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_concat_final_eval(test_dataloader, downstream_models,
                                                                 loss_fn, cxr_labels, device,
                                                                                                                        n_samples=n_samples, confidence_level=confidence_level)
    return test_auroc, test_mcc, downstream_model, auc_intervals, mcc_intervals


def chexpert_avg_eval(models, config, device, downstream_model_paths, cxr_labels=None,
                      n_samples=1000, confidence_level=0.05):
    """
    Like chexpert_eval but computes average model performance
    """
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema','No Finding','Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT AVG EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )
    
    # Reload downstream model with best parameters according to validation AUROC
    downstream_models = []
    for i in range(len(models)):
        downstream_model = models[i]
        downstream_model.load_state_dict(torch.load(downstream_model_paths[i]))
        downstream_models.append(downstream_model)
        
    boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_avg_final_eval(test_dataloader, downstream_models,
                                                                 loss_fn, cxr_labels, device,
                                                                                                                        n_samples=n_samples, confidence_level=confidence_level)
    return auc_intervals, mcc_intervals


def chexpert_comparison_eval(model1, model2, config, device, downstream_model_path1, downstream_model_path2,
                             cxr_labels=None, n_samples=1000, confidence_level=0.05):
    """
    model: model loaded
    config: dictionary containing
        'batch_size', 'loss_fn', 'optimizer', 'train_percent', 'epochs', and 'momentum' (if SGD)
    device: Either torch.device('cuda:0') or torch.device('cpu')
    downstream_model_path: path to save best performing downstream model
    """
    # cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other',
    #               'Pneumonia', 'Pneumothorax', 'Support Devices']
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT COMPARISON EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )

    downstream_model1 = model1.to(device)
    downstream_model2 = model2.to(device)
    # Reload downstream model with best parameters according to validation AUROC
    downstream_model1.load_state_dict(torch.load(downstream_model_path1))
    downstream_model2.load_state_dict(torch.load(downstream_model_path2))
    test_auroc1, test_auroc2, test_mcc1, test_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_comparison_final_eval(test_dataloader, downstream_model1,
                                                                                          downstream_model2, loss_fn,
                                                                                          cxr_labels, device,
                                                                                          n_samples, confidence_level
                                                                                         )
    return test_auroc1, test_auroc2, test_mcc1, test_mcc2, auc_intervals, mcc_intervals


def chexpert_ensemble_comparison_eval(models1, models2, config, device, downstream_model_paths1, downstream_model_paths2,
                             cxr_labels=None, ensemble_fn='median', n_samples=1000, confidence_level=0.05):

    # cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other',
    #               'Pneumonia', 'Pneumothorax', 'Support Devices']
    assert len(models1) == len(models2)
    assert len(downstream_model_paths1) == len(downstream_model_paths2)
    assert len(models1) == len(downstream_model_paths1)
    
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT ENSEMBLE COMPARISON EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )

    # Reload downstream model with best parameters according to validation AUROC
    downstream_models1 = []
    downstream_models2 = []
    for i in range(len(models1)):  # models1 and models2 must be the same length
        downstream_model1 = models1[i]  # don't copy to CUDA yet
        downstream_model1.load_state_dict(torch.load(downstream_model_paths1[i]))
        downstream_models1.append(downstream_model1)
        
        downstream_model2 = models2[i]
        downstream_model2.load_state_dict(torch.load(downstream_model_paths2[i]))
        downstream_models2.append(downstream_model2)

    test_auroc1, test_auroc2, test_mcc1, test_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_ensemble_comparison_final_eval(test_dataloader,
                                                                                                   downstream_models1,
                                                                                                   downstream_models2,
                                                                                                   loss_fn,
                                                                                                   cxr_labels, device,
                                                                                                   ensemble_fn,
                                                                                                   n_samples,
                                                                                                   confidence_level
                                                                                                  )
    return test_auroc1, test_auroc2, test_mcc1, test_mcc2, auc_intervals, mcc_intervals


def chexpert_concat_comparison_eval(models1, models2, config, device,
                                    downstream_model_paths1, downstream_model_paths2,
                                    cxr_labels=None,
                                    n_samples=1000, confidence_level=0.05):

    # cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other',
    #               'Pneumonia', 'Pneumothorax', 'Support Devices']
    assert len(models1) == len(models2)
    assert len(downstream_model_paths1) == len(downstream_model_paths2)
    assert len(models1) == len(downstream_model_paths1)
    
    if cxr_labels is None:
        cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
    batch_size = config['batch_size']
    train_percent = config.get('train_percent')
    K = config.get('train_K')
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    pretrained = config['pretrained']
    
    print("RUN CHEXPERT CONAT COMPARISON EVAL", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )

    # Reload downstream model with best parameters according to validation AUROC
    downstream_models1 = []
    downstream_models2 = []
    for i in range(len(models1)):  # models1 and models2 must be the same length
        downstream_model1 = models1[i]  # don't copy to CUDA yet
        downstream_model1.load_state_dict(torch.load(downstream_model_paths1[i]))
        downstream_models1.append(downstream_model1)
        
        downstream_model2 = models2[i]
        downstream_model2.load_state_dict(torch.load(downstream_model_paths2[i]))
        downstream_models2.append(downstream_model2)

    test_auroc1, test_auroc2, test_mcc1, test_mcc2, boot_auc_stats, boot_mcc_stats, auc_intervals, mcc_intervals = downstream_concat_comparison_final_eval(test_dataloader,
                                                                                                   downstream_models1,
                                                                                                   downstream_models2,
                                                                                                   loss_fn,
                                                                                                   cxr_labels, device,
                                                                                                   n_samples,
                                                                                                   confidence_level
                                                                                                  )
    return test_auroc1, test_auroc2, test_mcc1, test_mcc2, auc_intervals, mcc_intervals
