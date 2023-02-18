import sys, os

pypath = 'tuning'
sys.path.append(pypath)

for dir_name in os.listdir(pypath):
    dir_path = os.path.join(pypath, dir_name)
    if os.path.isdir(dir_path):
        print(dir_path)
        sys.path.append(dir_path)

import cxrlearn
import torch
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Process parameters to be used during few shot.')

# Training hyperparams
parser.add_argument("--optimizer", type=str, default="adam",
                help="The optimizer to use for training.")
parser.add_argument('--batch_size', type=int, default=32,
                help='Batch size to be used for training.')
parser.add_argument('--epochs', type=int, default=100,
                help='Number of epochs to train model for training.')
parser.add_argument('--lr', type=float, default=5e-4,
                help='Learning rate to be used for training.')
parser.add_argument('--epsilon', type=float, default=1e-8,
                help='Epsilon to be used for training.')
parser.add_argument('--few_shot', action="store_true",
                help='Perform few shot learning.')
parser.add_argument('--num_shot', type=int, default=5,
                help='Number of shots to be used. A value of 5 would lead to 5-shot learning. Only valid if --few_shot tag is used.')
parser.add_argument('--percent', type=float, default=0.01,
                help='Percent of training data to be used. A value of 0.01 would lead to 1% being used. Only valid if --few_shot tag is NOT used.')
parser.add_argument("--base_path", type=str, default="/deep/group/data/med-data",
                help="Base path for h5 and csv files for test/val/train files, named test_cxr.h5 final_paths.csv train.h5 train.csv valid.h5 valid.csv")

# Model hyperparams
parser.add_argument('--use_pretrained', action="store_true",
                help='Use a pretrained version of the model.')
parser.add_argument('--use_lineval', action="store_true",
                help='Perform linear evaluation (rather than finetuning) of the model.')
parser.add_argument('--downstream_path', type=str,
                help='Path where downstream models can be stored.')
parser.add_argument('--model_type', type=str,
                help='Type of model to be evaluated.')
parser.add_argument('--num_runs', type=int, default=10,
                help='Number of model runs to be evaluated.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(args, num_classes):
    if args.model_type == 'dracon':
        return cxrlearn.dracon(device=device,freeze_backbone=args.use_lineval,num_out=num_classes)
    elif args.model_type == 'chexzero':
        return cxrlearn.chexzero(device=device,freeze_backbone=args.use_lineval,linear_layer_dim=512,num_out=num_classes)
    elif args.model_type == 'convirt':
        return cxrlearn.convirt(device=device,freeze_backbone=args.use_lineval,num_out=num_classes)
    elif args.model_type == 'gloria':
        return cxrlearn.gloria(model="resnet50",device=device,freeze_backbone=freeze_backbone,num_ftrs=2048,num_out=num_classes)
    elif args.model_type == 'medaug':
        return cxrlearn.medaug(model="resnet50", pretrained_on="mimic-cxr",
                        device=device, freeze_backbone=args.use_lineval, num_out=num_classes)
    elif args.model_type == 'mococxr':
        return cxrlearn.mococxr(model="resnet50",device=device,freeze_backbone=args.use_lineval,num_out=num_classes)
    elif args.model_type == 's2mts2':
        return cxrlearn.s2mts2(device=device,freeze_backbone=args.use_lineval,num_out=num_classes)
    elif args.model_type == 'resnet':
        return cxrlearn.resnet(device=device, num_out=num_classes)
    else:
        print("ERROR: Please make sure to enter a valid model name!")
        return None

def execute_few_shot():
    args = parser.parse_args()

    # Config
    if args.few_shot:
        config = dict(
            batch_size=args.batch_size,
            loss_fn=torch.nn.CrossEntropyLoss(),
            lr=args.lr,
            optimizer=args.optimizer,
            train_K=args.num_shot,
            epochs=args.epochs,
            base_path=args.base_path,
            pretrained=args.use_pretrained,
            eps=args.epsilon,
        )
    else:
        config = dict(
            batch_size=args.batch_size,
            loss_fn=torch.nn.CrossEntropyLoss(),
            lr=args.lr,
            optimizer=args.optimizer,
            train_percent=args.percent,
            epochs=args.epochs,
            base_path=args.base_path,
            pretrained=args.use_pretrained,
            eps=args.epsilon,
        )

    cxr_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'No Finding', 'Pleural Effusion']
    num_classes = len(cxr_labels)

    #1. Load PreTrained Model

    freeze_backbone = args.use_lineval

    final_eval=True
    num_runs = args.num_runs

    downstream_model_paths = []
    models = []

    #2. Finetune the model (or linear eval if freeze_backbone==True). Also loads the CheXpert dataset.
    #   And performs evaluation on test set if final_eval=True.
    for i in range(num_runs):
        model = get_model(args, num_classes)
        if model is None:
            return
        downstream_model_path = f"{args.downstream_path}_{i}.pt"
        cxrlearn.chexpert_experiment(model, config, device, downstream_model_path, final_eval=final_eval, cxr_labels=cxr_labels)

        downstream_model_paths.append(downstream_model_path)
        models.append(model)

    #3. Perform evaluation.
    auroc, mcc, _, auc_intervals, mcc_intervals = cxrlearn.chexpert_concat_evaluate(models, config, device,
                                                                             downstream_model_paths,
                                                                             cxr_labels=cxr_labels,
                                                                             n_samples=1000,
                                                                             confidence_level=0.05)
    save_dict = {
        "config": config,
        "auroc": auroc,
        "mcc": mcc,
        "auc_intervals": auc_intervals,
        "mcc_intervals": mcc_intervals
    }
    downstream_pickle_path = f"{args.downstream_path}_eval_dict.pkl"
    pickle.dump(save_dict, open(downstream_pickle_path, "wb"))

    print(save_dict)

if __name__ == "__main__":
    execute_few_shot()
