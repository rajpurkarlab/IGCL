import torch
import os
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
from torch import nn
from torch import optim
from sklearn.metrics import roc_auc_score
import warnings



def _finetune(train_dataset=None,val_dataset=None,model=None,device=torch.device("cpu"),optimizer=optim.SGD,scheduler=optim.lr_scheduler.CosineAnnealingLR,scheduler_stepping=200,batch_size=32,epochs=100,lr=0.001,momentum=0.9,shuffle=True,num_workers=4,ckpt_path="./pt-finetune"):
    if device==torch.device("cpu"):
        warnings.warn("Finetuning on CPU.... Use GPU if available for faster training! pass device variable in chexzero function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    if val_dataset==None:
        warnings.warn("Validation set is None, it is recommended to use a validation set!")

    assert train_dataset != None, "train dataset object passed is None."
    assert model != None, "model object passed is None"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"
    assert scheduler_stepping>0, "scheduler_stepping should be greater than 0!"
    assert batch_size>0, "batch_size should be greater than 0!"
    assert epochs>0, "epochs should be greater than 0!"
    assert lr>0, "lr should be greater than 0!"
    assert momentum>0, "momentum should be greater than 0!"
    assert num_workers>0, "num_workers should be greater than 0!"
    assert isinstance(shuffle,bool), "shuffle should be a True or False boolean"
    assert isinstance(ckpt_path,str), "ckpt_path has to be a string"

    if not os.path.exists(ckpt_path):
        print(str(ckpt_path) + " does not exist. Creating the path....")
        os.mkdir(str(ckpt_path))

    bin_info = str(train_dataset.name.split("-")[-1])
    prediction_type = bin_info

    if prediction_type=="binary":
        criterion = nn.BCEWithLogitsLoss()
    elif prediction_type=="multi":
        criterion = nn.CrossEntropyLoss()
    else:
        raise "dataloader component isnt processed through cxrlearn.cxr_to_pt function"




    optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    scheduler = scheduler(optimizer, scheduler_stepping)

    model.train()

    steps = 0
    batch = 0

    running_loss = 0
    best_val_loss = 1000
    best_val_auc = 0
    best_epoch = 0
    early_stop_epochs = 15
    train_losses, val_losses, val_aucs = [], [], []

    print("-------------------------------------")
    print("Training is starting..")
    print("-------------------------------------")
    for epoch in range(epochs):
        steps = 0
        for inputs, labels in tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True)):
            steps += 1
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += labels.size()[0]
            # if steps % epochs/4 == 0:
            #     scheduler.step()

        if val_dataset!=None:
            val_loss = 0
            accuracy = 0
            model.eval()
            preds_v = []
            true_v = []
            with torch.no_grad():
                for inputs_val, labels_val in tqdm(DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=True)):
                    inputs_val, labels_val = inputs_val.to(torch.float32).to(device), labels_val.to(torch.float32).to(device)
                    logps = model.forward(inputs_val)
                    batch_loss = criterion(logps, labels_val)
                    val_loss += batch_loss.item()
                    preds_v.append(logps)
                    true_v.append(labels_val)
            val_preds = torch.cat(preds_v).cpu()
            val_labels = torch.cat(true_v).cpu()

            if prediction_type=="binary":
                results = roc_auc_score(val_labels, val_preds, average="weighted")
            elif prediction_type=="multi":
                results = roc_auc_score(val_labels, val_preds, average="weighted",multi_class='ovr')


            train_losses.append(running_loss/batch)
            val_losses.append(val_loss/len(val_dataset))
            val_aucs.append(results.mean())

            if val_loss/len(val_dataset) < best_val_loss:
                best_val_loss = val_loss/len(val_dataset)
            if results.mean() > best_val_auc:
                best_val_auc = results.mean()
                best_epoch = epoch
                torch.save(model,ckpt_path+"/"+str(model.name)+"_best.pt")

        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {100*running_loss/batch:.4f}.. ")


        if val_dataset!=None:
            print(f"Val loss: {100*val_loss/len(val_dataset):.4f}.. "
            f"Val AUC: {round(results.mean(), 5)}.. ")

            print(f"Best Val loss: {100*best_val_loss:.4f}.. "
                f"Best Val AUC: {round(best_val_auc, 5)}.."
                f"Last LR: {scheduler.get_last_lr()[0]:.9f}..")

        running_loss = 0
        batch = 0
        model.train()

    print("-------------------------------------")
    print("..Training is done!")

    if val_dataset!=None:
        print("..Best Epoch: ", best_epoch + 1)

    torch.save(model, ckpt_path+"/"+str(model.name)+"_final.pt")
