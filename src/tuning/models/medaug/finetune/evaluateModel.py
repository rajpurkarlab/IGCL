import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


df_preds = pd.read_csv("/deep/group/cxr-transfer/SSL-methods/MedAug/experiments/RANZCR-SSL-Linear/results/test/predictions.csv")
df_gts = pd.read_csv("/deep/group/cxr-transfer/SSL-methods/MedAug/experiments/RANZCR-SSL-Linear/results/test/groundtruth.csv")

attrs = list(df_gts.head(0))


aucs = []
for i in attrs:
    pred = (df_preds[i].values)
    gt = (np.int32(df_gts[i].values))
#     print((gt))
    ns_auc = roc_auc_score(gt,pred)
    print("------------"+i+" AUCROC = "+str(ns_auc)+"------------")
#     print(ns_auc)
    aucs.append(ns_auc)
print("---------------- Average AUCROC = "+str(np.mean(aucs))+"-----------------")