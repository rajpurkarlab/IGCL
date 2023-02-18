from gc import freeze
# from datasets.chexpert_dataset import _load_chexpert

from datasets.prepare_dataset import _cxr_to_pt

from tuning.models.gloria.gloria_cxrt import _gloria
from tuning.models.refers.refers_cxrt import _refers
from tuning.models.s2mts2.s2mts2_cxrt import _s2mts2
from tuning.models.convirt.convirt_cxrt import _convirt
from tuning.models.chexzero.chexzero_cxrt import _chexzero
from tuning.models.medaug.medaug_cxrt import _medaug
from tuning.models.mococxr.mococxr_cxrt import _mococxr
from tuning.models.resnet.resnet_sl import _resnet

from tuning.models.dracon.dracon_cxrt import _dracon
from tuning.models.rgcn.rgcn_cxrt import _rgcn

from tuning.train.chexpert_finetuning import run_chexpert_experiment
from tuning.train.finetuning import _finetune
from tuning.test.chexpert_eval import chexpert_eval, chexpert_ensemble_eval
from tuning.test.chexpert_eval import chexpert_comparison_eval, chexpert_ensemble_comparison_eval
from tuning.test.chexpert_eval import chexpert_concat_comparison_eval, chexpert_concat_eval
from tuning.test.chexpert_eval import chexpert_avg_eval
from tuning.test.eval import _evaluate, _evaluate_fromckpts

import torch
from torch import optim

def cxr_to_pt(csv_file,path_col,class_col,dataset_name,out_pt_address,reshape_size=(224,224),pt_withoutLabels=False,prediction_type="binary",skip_loading=False,fewshot_class=None,fewshot_perclass=0,save_data=True):
    return _cxr_to_pt(csv_file,path_col,class_col,dataset_name,out_pt_address,reshape_size=reshape_size,pt_withoutLabels=pt_withoutLabels,prediction_type=prediction_type,skip_loading=skip_loading,fewshot_class=fewshot_class,fewshot_perclass=fewshot_perclass,save_data=save_data)
    
def chexzero(device=torch.device("cpu"),freeze_backbone=True,linear_layer_dim=512,num_out=1):
    return _chexzero(device=device,freeze_backbone=freeze_backbone,linear_layer_dim=linear_layer_dim,num_out=num_out)

def convirt(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    return _convirt(device=device,freeze_backbone=freeze_backbone,num_out=num_out)

def gloria(model="resnet50",device=torch.device("cpu"),freeze_backbone=True,num_ftrs=2048,num_out=1):
    return _gloria(model=model,device=device,freeze_backbone=freeze_backbone,num_ftrs=num_ftrs,num_out=num_out)

def medaug(model="resnet50",pretrained_on="mimic-cxr",device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    return _medaug(model=model,pretrained_on=pretrained_on,device=device,freeze_backbone=freeze_backbone,num_out=num_out)

def mococxr(model="resnet50",device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    return _mococxr(model=model,device=device,freeze_backbone=freeze_backbone,num_out=num_out)

def refers(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    return _refers(device=device,freeze_backbone=freeze_backbone,num_out=num_out)

def s2mts2(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    return _s2mts2(device=device,freeze_backbone=freeze_backbone,num_out=num_out)

def resnet(device=torch.device("cpu"),num_out=1):
    return _resnet(device=device,num_out=num_out)

def finetune(train_dataset=None,val_dataset=None,model=None,device=torch.device("cpu"),optimizer=optim.SGD,scheduler=optim.lr_scheduler.CosineAnnealingLR,scheduler_stepping=200,batch_size=32,epochs=100,lr=0.001,momentum=0.9,shuffle=True,num_workers=4,ckpt_path="./pt-finetune"):
    return _finetune(train_dataset=train_dataset,val_dataset=val_dataset,model=model,device=device,optimizer=optimizer,scheduler=scheduler,scheduler_stepping=scheduler_stepping,batch_size=batch_size,epochs=epochs,lr=lr,momentum=momentum,shuffle=shuffle,num_workers=num_workers,ckpt_path=ckpt_path)

def chexpert_experiment(model, config, device, downstream_model_path,
                        final_eval=False, upstream_config=None, cxr_labels=None):
    return run_chexpert_experiment(model, config, device, downstream_model_path, final_eval=final_eval,
                                   upstream_config=upstream_config, cxr_labels=cxr_labels)

def evaluate(test_dataset=None,model=None,device=torch.device("cpu")):
    return _evaluate(test_dataset=test_dataset,model=model,device=device)

def evaluate_fromckpts(test_dataset=None,ckpt_path=None,device=torch.device("cpu")):
    return _evaluate_fromckpts(test_dataset=test_dataset,ckpt_path=ckpt_path,device=device)


def chexpert_evaluate(model, config, device, downstream_model_path, cxr_labels=None,
                      n_samples=1000, confidence_level=0.05):
    return chexpert_eval(model, config, device, downstream_model_path, cxr_labels=cxr_labels,
                         n_samples=n_samples, confidence_level=confidence_level)

def chexpert_ensemble_evaluate(models, config, device, downstream_model_paths,
                               cxr_labels=None, ensemble_fn='median', n_samples=1000, confidence_level=0.05):
    return chexpert_ensemble_eval(models, config, device, downstream_model_paths, cxr_labels=cxr_labels,
                                  ensemble_fn=ensemble_fn,
                                  n_samples=n_samples, confidence_level=confidence_level
                                 )


def chexpert_concat_evaluate(models, config, device, downstream_model_paths, cxr_labels=None,
                             n_samples=1000, confidence_level=0.05):
    return chexpert_concat_eval(models, config, device, downstream_model_paths,
                                    cxr_labels=cxr_labels,
                                    n_samples=n_samples, confidence_level=confidence_level)


def chexpert_avg_evaluate(models, config, device, downstream_model_paths, cxr_labels=None,
                          n_samples=1000, confidence_level=0.05):
    return chexpert_avg_eval(models, config, device, downstream_model_paths, cxr_labels=cxr_labels,
                      n_samples=n_samples, confidence_level=confidence_level)


def chexpert_comparison_evaluate(model1, model2, config, device, downstream_model_path1,
                                 downstream_model_path2, cxr_labels=None,
                                 n_samples=1000, confidence_level=0.05,
                                ):
    return chexpert_comparison_eval(model1, model2, config, device, downstream_model_path1,
                                    downstream_model_path2, cxr_labels=cxr_labels,
                                    n_samples=n_samples, confidence_level=confidence_level,
                                   )

def chexpert_ensemble_comparison_evaluate(models1, models2, config, device, downstream_model_paths1,
                                          downstream_model_paths2, cxr_labels=None,
                                          ensemble_fn='median', n_samples=1000, confidence_level=0.05):
    return chexpert_ensemble_comparison_eval(models1, models2, config, device, downstream_model_paths1,
                                             downstream_model_paths2, cxr_labels=cxr_labels,
                                             ensemble_fn=ensemble_fn,
                                             n_samples=n_samples, confidence_level=confidence_level)


def chexpert_concat_comparison_evaluate(models1, models2, config, device,
                                    downstream_model_paths1, downstream_model_paths2,
                                    cxr_labels=None,
                                    n_samples=1000, confidence_level=0.05):
    return chexpert_concat_comparison_eval(models1, models2, config, device,
                                           downstream_model_paths1, downstream_model_paths2,
                                           cxr_labels=cxr_labels,
                                           n_samples=n_samples, confidence_level=confidence_level)
####
def dracon(device=torch.device("cpu"), freeze_backbone=True, num_out=1):
    return _dracon(device=device, freeze_backbone=freeze_backbone, num_out=num_out)

def rgcn(device=torch.device("cpu"), freeze_backbone=True, num_out=1):
    return _rgcn(device=device, freeze_backbone=freeze_backbone, num_out=num_out)