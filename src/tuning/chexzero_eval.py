import cxrlearn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config
config = dict(
    batch_size=32,
    loss_fn=torch.nn.CrossEntropyLoss(),
    lr=1e-4,
    optimizer='adam',
    train_percent=0.01,
    epochs=100,

    pretrained=True,  # since config.pretrained is True
    eps=1e-8,
)
cxr_labels = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema','No Finding','Pleural Effusion']
num_classes = len(cxr_labels)


#1. Load PreTrained Model

freeze_backbone = True  # if this is true, only linear eval
model = cxrlearn.medaug(model="resnet50", pretrained_on="mimic-cxr",
                        device=device, freeze_backbone=freeze_backbone, num_out=num_classes)

# model = cxrlearn.chexzero(device=device,freeze_backbone=freeze_backbone,linear_layer_dim=512,num_out=num_classes)
# model = cxrlearn.refers(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.convirt(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.gloria(model="resnet50",device=device,freeze_backbone=freeze_backbone,num_ftrs=2048,num_out=num_classes)
# model = cxrlearn.s2mts2(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.mococxr(model="resnet50",device=device,freeze_backbone=freeze_backbone,num_out=num_classes)

# model = cxrlearn.resnet(device=device,num_out=num_classes)

print('MODEL NAME', model.name)



#2. Finetune the model (or linear eval if freeze_backbone==True). Also loads the CheXpert dataset.
#   And performs evaluation on test set if final_eval=True.
final_eval=True
downstream_model_path = f'./CheXpert/{model.name}.pt'

cxrlearn.chexpert_experiment(model, config, device, downstream_model_path, final_eval=final_eval)

# #4. Evaluate just-trained model
# print("------------Result for Last Epoch-------------------")
# cxrlearn.evaluate(test_dataset=test,model=model,device=device)

# print("-------------------------")

print(model.name)
