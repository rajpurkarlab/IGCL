"""
Objects for training models.
"""

import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import logging
import torchvision


#from model.layer import FCLayer, MultiLabelBinaryFCLayer

logger = logging.getLogger('transfer')

def unpack_batch(batch):
    new = []
    for e in batch:
        new.append(e.cuda())
    return new

def resnet50(pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    dim_feats = model.fc.in_features
    return model, dim_feats

def set_classifier(model, new_clf_layer):
    """
    Set the last classifier layer in the model to a new classifier layer.
    """
    if hasattr(model, 'classifier'):
        # assert model.classifier.in_features == new_clf_layer.in_features, \
            # "in_features does not match between old and new classifier layer."
        model.classifier = new_clf_layer
    elif hasattr(model, 'fc'):
        # assert model.fc.in_features == new_clf_layer.in_features, \
            # "in_features does not match between old and new classifier layer."
        model.fc = new_clf_layer
    else:
        raise Exception(f"Cannot set classifier layer for model: neither 'fc' nor 'classifier' layer exists.")

def get_classifier(model):
    if hasattr(model, 'classifier'):
        return model.classifier
    elif hasattr(model, 'fc'):
        return model.fc
    else:
        raise Exception(f"Cannot find classifier layer for model: neither 'fc' nor 'classifier' layer exists.")

        
class FCLayer(nn.Module):
    """
    A basic FC classifier layer with `num_class` output classes.
    """
    def __init__(self, in_features, num_class, dropout=None):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_features, num_class)
        self.in_features = in_features
        self.out_features = num_class
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            return self.fc(self.dropout(x))
        else:
            return self.fc(x)

class MultiLabelBinaryFCLayer(nn.Module):
    """
    A multi-label FC classifier layer with `num_label` output heads and each a binary output.
    """
    def __init__(self, in_features, num_label, dropout=None):
        super(MultiLabelBinaryFCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features, num_label), nn.Sigmoid())
        self.in_features = in_features
        self.out_features = num_label
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            return self.fc(self.dropout(x))
        else:
            return self.fc(x)

        
def get_weighted_cross_entropy_loss(all_labels=None, log_dampened=False):
    if all_labels is None:
        return nn.CrossEntropyLoss()
    
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)
    _, weights = np.unique(all_labels, return_counts=True)
    weights = weights / float(np.sum(weights))
    weights = 1. / weights
    if log_dampened:
        weights = 1 + np.log(weights)
    weights /= np.sum(weights) # normalize
    crit = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).type('torch.FloatTensor')
    )
    return crit
        
class ImageTrainer(object):
    def __init__(self, opt, train_labels=None, multilabel=False, multiclass=False, random_init=False):
        self.opt = opt
        assert not (multiclass and multilabel), "Trainer does not support both multiclass and multilabel are True."
        self.multilabel = multilabel
        self.multiclass = multiclass
        
        # initialize model and crit
        pretrained = not random_init
        self.model, dim_feats = resnet50()
        # reset model classifier
        if multilabel:
            self._num_class = opt['num_label']
            set_classifier(self.model, MultiLabelBinaryFCLayer(
                in_features=dim_feats,
                num_label=self._num_class,
                dropout=opt['dropout'])
            )
            self.crit = nn.BCELoss()
        else:
            self._num_class = opt['num_class']
            set_classifier(self.model, FCLayer(
                in_features=dim_feats,
                num_class=self._num_class,
                dropout=opt['dropout'])
            )
            self.crit = get_weighted_cross_entropy_loss(train_labels, opt['log_dampened'])
        self.model.train()#.cuda()
        self.crit#.cuda()

        # optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
        self._clf = get_classifier(self.model)
        self.clf_optimizer = optim.Adam(self._clf.parameters(), lr=opt['clf_lr'])
    
    def freeze_all_but_clf(self):
        logger.info("Freezing all weights of image encoder; only classifier layer will be trained.")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self._clf.parameters():
            param.requires_grad = True
    
    def update(self, batch, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        imgs, labels = unpack_batch(batch)
        
        # step forward
        self.model.train()
        optimizer.zero_grad()
        output = self.model(imgs)
        loss = self.crit(output, labels)

        # backward
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, batch):
        imgs, labels = unpack_batch(batch)
        self.model.eval()
        logits = self.model(imgs)
        loss = self.crit(logits, labels)
        if self.multilabel:
            probs = logits.data.cpu().numpy().tolist()
        else:
            probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        return probs, loss.item()

    def get_metrics(self, y_true, y_pred, label_names=None):
        if self.multilabel:
            assert label_names is not None, \
                "Label names must be provided for metrics calculation in multilabel model."
            return self._get_metrics_multi_label(y_true, y_pred, label_names)
        elif self.multiclass:
            return self._get_metrics_multi_class(y_true, y_pred)
        else:
            return self._get_metrics_single_class_single_label(y_true, y_pred)
    
    def _get_metrics_single_class_single_label(self, y_true, y_pred):
        auc = scorer.get_auc_score(y_true, y_pred)
        f1  = scorer.get_f1_score(y_true, y_pred)
        metrics = {'auc': auc, 'f1': f1}
        return metrics
    
    def _get_metrics_multi_class(self, y_true, y_pred):
        f1 = scorer.get_f1_score(y_true, y_pred, average='macro')
        accu = scorer.get_accuracy(y_true, np.argmax(y_pred, axis=1))
        metrics = {'f1': f1, 'accuracy': accu}
        return metrics
    
    def _get_metrics_multi_label(self, y_true, y_pred, label_names):
        assert y_true.shape[1] == y_pred.shape[1] == len(label_names), \
            "Number of labels must match in metrics calculation."
        metrics = dict()
        avg_auc = 0
        avg_f1 = 0
        for i, label in enumerate(label_names):
            auc = scorer.get_auc_score(y_true[:,i], y_pred[:,i])
            f1 = scorer.get_f1_score(y_true[:,i], y_pred[:,i])
            metrics[f'auc:{label.lower()}'] = auc
            metrics[f'f1:{label.lower()}'] = f1
            avg_auc += auc
            avg_f1 += f1
        metrics['auc'] = avg_auc / len(label_names)
        metrics['f1'] = avg_f1 / len(label_names)
        return metrics
    
    def get_metrics_by_coalesce(self, y_pred, y_group, group2label):
        """
        A metrics calculator that aggregate predictions across examples within a group.
        This is specifically created for the MURA dataset, with calculates AUC at the study level.
        """
        # coalesce prob outputs within a group
        group2prob = scorer.coalesce_predictions(y_pred, y_group)
        y_prob, y_true = [], []
        for g in group2prob:
            y_prob.append(group2prob[g])
            y_true.append(group2label[g])
        auc = scorer.get_auc_score(y_true, y_prob)
        metrics = {'auc': auc}
        return metrics
    
    def pretty_print_metrics(self, metrics, label_names):
        assert self.multilabel, "pretty_print_metrics only works in multilabel setting."
        out_str = "{:20}\t{:5}\t{:5}\n".format("Variable", "F1", "AUC")
        for l in label_names:
            out_str += "{:20}\t{:.2f}\t{:.2f}\n".format(
                l,
                metrics[f"f1:{l.lower()}"]*100,
                metrics[f"auc:{l.lower()}"]*100
            )
        out_str += "{:20}\t{:.2f}\t{:.2f}\n".format(
            "Average",
            metrics["f1"]*100,
            metrics["auc"]*100
        )
        return out_str
    
    def pretty_print_auc_from_metrics(self, metrics):
        assert self.multilabel, "pretty_print_metrics only works in multilabel setting."
        names = sorted(list(filter(lambda x: x.lower().startswith('auc'), metrics.keys())))
        vals = ["{:.2f}".format(metrics[n]*100) for n in names]
        names = [x.replace(' ', '_') for x in names] # replace space in names
        out_str = " ".join(names) + "\n" + " ".join(vals)
        return out_str
    
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt
        }
        try:
            torch.save(params, filename)
            logger.info(f"Model saved to {filename}.")
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")
    
    @staticmethod
    def load_from_file(filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception(f"Cannot load model from {filename}")
            sys.exit(1)
        opt = checkpoint['config']
        model = checkpoint['model']
        return model , opt