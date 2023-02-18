import cxrlearn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Declare CSVssftp://adgulati@171.64.68.31/deep2/group/cxr-transfer/datasets/ranzcr-clip-catheter-line-classification/train.csv
train_csv = "/deep2/group/cxr-transfer/datasets/ranzcr-clip-catheter-line-classification/train.csv"
val_csv = "/deep2/group/cxr-transfer/datasets/ranzcr-clip-catheter-line-classification/valid.csv"
test_csv = "/deep2/group/cxr-transfer/datasets/ranzcr-clip-catheter-line-classification/test.csv"

#Declare path_col, target_col, weights_addrs

ft_pt_address = "./pt-finetune/"


path_col = ["StudyInstanceUID"]

class_cols = ["Cardiomegaly","Spondylosis","Hernia and Hiatal","Airspace Disease","Pleural Effusion","Costophrenic Angle","Emphysema","Pulmonary Atelectasis","Medical Device","Scoliosis","Nodule","Atherosclerosis","Calcinosis","Pneumonia","Granuloma","Calcified Granuloma","Pulmonary Congestion","Opacity","Pulmonary Edema","normal"]
#class_cols = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Lesion","Lung Opacity","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]


out_pt_address = "./pt-dataset/"

num_classes=len(class_cols)

#1. Setup Dataset


train = cxrlearn.cxr_to_pt(train_csv,path_col,class_cols,"ranzcr-train",out_pt_address)
val = cxrlearn.cxr_to_pt(val_csv,path_col,class_cols,"ranzcr-val",out_pt_address)
test = cxrlearn.cxr_to_pt(test_csv,path_col,class_cols,"ranzcr-test",out_pt_address)


#2. Load PreTrained Model

freeze_backbone = False

# model = cxrlearn.medaug(model="resnet50",pretrained_on="mimic-cxr",device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.chexzero(device=device,freeze_backbone=freeze_backbone,linear_layer_dim=512,num_out=num_classes)
# model = cxrlearn.refers(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.convirt(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.gloria(model="resnet50",device=device,freeze_backbone=freeze_backbone,num_ftrs=2048,num_out=num_classes)
# model = cxrlearn.s2mts2(device=device,freeze_backbone=freeze_backbone,num_out=num_classes)
# model = cxrlearn.mococxr(model="resnet50",device=device,freeze_backbone=freeze_backbone,num_out=num_classes)

model = cxrlearn.resnet(device=device,num_out=num_classes)



model.name = train.name+"-"+model.name


#3. FineTune the CNN/only-Linear layer
cxrlearn.finetune(train_dataset=train,val_dataset=val,model=model,device=device,batch_size=32,epochs=100,lr=0.001,momentum=0.9)


# #4. Evaluate just-trained model
# print("------------Result for Last Epoch-------------------")
# cxrlearn.evaluate(test_dataset=test,model=model,device=device)

# print("-------------------------")

#4. Or Load weights and evaluate!
print("------------Result for Best Epoch-------------------")
cxrlearn.evaluate_fromckpts(test_dataset=test,ckpt_path="./pt-finetune/"+str(model.name)+"_best.pt",device=device)

print(model.name)
