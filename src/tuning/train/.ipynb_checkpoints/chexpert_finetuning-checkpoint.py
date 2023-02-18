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
from sklearn.metrics import average_precision_score
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
    
    dataframes = []
    
    radiologists = pd.read_csv('radiologists.csv')
    
    for i in range(num_classes): 
#         print('{}.'.format(cxr_labels[i]))
        y_pred_i = y_pred[:, i] # (num_samples,)
        y_true_i = y_true[:, i] # (num_samples,)
        
        cxr_label = cxr_labels[i]
        
        rad = radiologists[cxr_label]
        xs = rad[:3]
        ys = rad[3:]
        
        ''' ROC CURVE '''
        roc_name = cxr_labels[i] + ' ROC Curve'
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot=plot, rad_xs=xs, rad_ys=ys)
        
#         print('AUROC: '.format(cxr_labels[i]) + str(roc_auc))
        sens, spec = choose_operating_point(fpr, tpr, thresholds)
#         print('Sensitivity: ' + str(sens))
#         print('Specificity: ' + str(spec))

        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
        dataframes.append(df)
        
        ''' PRECISION-RECALL CURVE '''
        pr_name = cxr_labels[i] + ' Precision-Recall Curve'
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name, plot=plot)
        
#         print('\n')
    #     p = compute_precision(test_y, test_pred)
    #     print('Average Precision: ' + str(p))
    dfs = pd.concat(dataframes, axis=1)
    return dfs

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
    
def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000): 
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
    
    boot_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        sample_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

    
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


def downstream_train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        # print("TYPE OF X downstream", type(X), X.type())
        pred = model(X)
        # print("TRAIN PRED", pred.shape, pred)
        loss = loss_fn(pred, y)
        # print("TRAIN LOSS", loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def downstream_test_loop(dataloader, model, loss_fn, cxr_labels, device):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    y_pred = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            # print("PRED", pred.shape, pred)
            test_loss += loss_fn(pred, y).item()
            y_pred.extend(pred.to('cpu').numpy())
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss /= size
    
    y_pred, y_actual = np.array(y_pred), np.array(y_actual)
    # print("Y_PRED", y_pred)
    # print("Y_ACTUAL", y_actual)
    auc_df = evaluate(y_pred, y_actual, cxr_labels, plot=False).astype(float)
    print(auc_df)
    mean_auroc = auc_df.mean(axis=1)[0]
    print(f"Val Error: \n Mean AUROC: {mean_auroc:>4f}, Avg loss: {test_loss:>8f} \n")
    return mean_auroc, auc_df
    

def downstream_final_eval(dataloader, model, loss_fn, cxr_labels, device):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    y_pred = []
    y_actual = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_pred.extend(pred.to('cpu').numpy())
            y_actual.extend(y.to('cpu').numpy())
    
    test_loss /= size
    
    y_pred, y_actual = np.array(y_pred), np.array(y_actual)
    auc_df = evaluate(y_pred, y_actual, cxr_labels, plot=True).astype(float)
    print(auc_df)
    mean_auroc = auc_df.mean(axis=1)[0]
    print(f"Test Error: \n Mean AUROC: {mean_auroc:>4f}, Avg loss: {test_loss:>8f} \n")
    
    boot_stats, intervals_df = bootstrap(y_pred, y_actual, cxr_labels, n_samples=100)
    print(boot_stats)
    print(intervals_df)
    return mean_auroc, boot_stats, intervals_df


def run_chexpert_experiment(model, config, device, downstream_model_path, final_eval=False,
                            upstream_config=None, cxr_labels=None):
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
    
    print("RUN CHEXPERT EXPERIMENT", K, epochs, loss_fn, pretrained)
    
    train_dataloader, val_dataloader, test_dataloader = load_chexpert(cxr_labels,
                                                                      batch_size,
                                                                      pretrained,
                                                                      train_percent,
                                                                      K,
                                                                     )
    # print(f'len train loader: {len(train_dataloader)}, len val loader: {len(val_dataloader)}, len test loader: {len(test_dataloader)}')
    # if linear:
    #     downstream_model = LinearModel(model, len(cxr_labels)).to(device)
    # else:
    #     downstream_model = FinetunedModel(model, len(cxr_labels)).to(device)
    downstream_model = model.to(device)
        
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(downstream_model.parameters(), lr=config['lr'], momentum=config['momentum'])
    else:
        optimizer = torch.optim.Adam(downstream_model.parameters(), lr=config['lr'], eps=config['eps'])
        
    highest_mean_auroc = 0.
    best_auc_df = pd.DataFrame()
    for t in range(config['epochs']):
        print(f"Downstream Epoch {t+1}\n-------------------------------")
        downstream_train_loop(train_dataloader, downstream_model, loss_fn, optimizer, device)
        mean_auroc, auc_df = downstream_test_loop(val_dataloader, downstream_model, loss_fn, cxr_labels, device)
        if mean_auroc > highest_mean_auroc:
            print('New highest downstream val AUROC')
            save(downstream_model, downstream_model_path)
            highest_mean_auroc = mean_auroc
            best_auc_df = auc_df
    print("Done with CheXpert training!")
    print("HIGHEST MEAN VAL AUROC:", highest_mean_auroc)
    print("BEST AUC DATAFRAME:", best_auc_df)
    
    if final_eval:
        # Reload downstream model with best parameters according to validation AUROC
        downstream_model.load_state_dict(torch.load(downstream_model_path))
        test_auroc, boot_stats, intervals_df = downstream_final_eval(test_dataloader, downstream_model,
                                                                     loss_fn, cxr_labels, device)
        return highest_mean_auroc, best_auc_df, downstream_model, intervals_df
    
    return highest_mean_auroc, best_auc_df, downstream_model