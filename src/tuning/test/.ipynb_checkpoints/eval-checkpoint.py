import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

def _eval(test_data,model,device,labels_available=True):

    if labels_available:
        preds = []
        true = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(DataLoader(test_data, batch_size=16)):
                inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
                logps = model.forward(inputs)
                preds.append(logps)
                true.append(labels)
        model.train()
        test_preds = torch.cat(preds).cpu()
        test_labels = torch.cat(true).cpu()

        return test_preds,test_labels
    else:
        preds = []
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(DataLoader(test_data, batch_size=16)):
                inputs = inputs.to(torch.float32).to(device)
                logps = model.forward(inputs)
                preds.append(logps)
        model.train()
        test_preds = torch.cat(preds).cpu()
        return test_preds


def _evaluate_fromckpts(test_dataset=None,ckpt_path=None,device=torch.device("cpu")):
    assert test_dataset != None, "train dataset object passed is None."
    assert os.path.exists(ckpt_path), str(ckpt_path) + " does not exist."
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"

    model = torch.load(ckpt_path)
    results = _evaluate(test_dataset=test_dataset,model=model,device=device)
    return results


def _evaluate(test_dataset=None,model=None,device=torch.device("cpu"),labels_available=True,save_results="./results/"):
    assert test_dataset != None, "train dataset object passed is None."
    assert model != None, "model object passed is None"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"

    if labels_available:
        test_preds, test_labels = _eval(test_dataset,model,device,labels_available=True)
    else:
        test_preds = _eval(test_dataset,model,device,labels_available=False)

    if labels_available:
        bin_info = str(test_dataset.name.split("-")[-1])
        prediction_type = bin_info
        if prediction_type=="binary":
            # print(test_labels)
            # print(test_preds)
            # exit()
            results = roc_auc_score(test_labels, test_preds, average=None)
            _print_results(test_dataset,results)
        elif prediction_type=="multi":
            results = roc_auc_score(test_labels, test_preds, average="weighted", multi_class='ovr')
            print(test_dataset.labels[0], " AUROC: ", round(results,4))
        else:
            raise "dataloader component isnt processed through cxrlearn.cxr_to_pt function"
        _save_results(test_dataset,test_preds,test_labels,save_results=save_results,name=model.name+"-")
        return results
    else:
        _save_results(test_dataset,test_preds,save_results=save_results,name=model.name+"-")

def _save_results(test_dataset,test_preds,test_labels=None,save_results="./results/",name=None):
    if test_labels!=None:
        gt = pd.DataFrame(test_labels)
        gt.to_csv(save_results+name+"groundtruth.csv")
    pred = pd.DataFrame(test_preds)
    pred.to_csv(save_results+name+"predictions.csv")

    return





def _print_results(data,results):
    if len(data.labels)>1:
        for i, label in enumerate(data.labels):
            print(label, " AUROC: ", round(results[i],4))
        print("--------------------------------------------")
        print("Average AUROC: ", round(results.mean(), 4))
    else:
        print(data.labels, " AUROC: ", round(results,4))
