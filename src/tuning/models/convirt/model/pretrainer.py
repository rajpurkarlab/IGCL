"""
Trainer functions for jointly pretraining image and text models.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from model.pretrain_model import JointBinaryModel, JointNCEModel
from utils import constant, torch_utils

logger = logging.getLogger('transfer')

def unpack_batch(batch):
    for key, vals in batch.items():
        for i in range(len(vals)):
            vals[i] = vals[i].cuda()
    return batch

class BinaryPretrainer():
    def __init__(self, opt):
        self.opt = opt

        self.model = JointBinaryModel(self.opt)
        self.crit = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.model.cuda()
        self.crit.cuda()

        self.optimizer = get_optimizer(
            self.opt['optim'],
            self.parameters,
            lr=self.opt['lr'],
            weight_decay=self.opt['weight_decay']
        )

        # amp training
        self.use_amp = self.opt['amp']
        if self.use_amp:
            self.scaler = GradScaler()
    
    def update(self, batch):
        tensorized = unpack_batch(batch)
        labels = tensorized.pop('label')[0]

        img = tensorized['image'][0]
        text_ids, text_attention_mask = tensorized['text']

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            logits = self.model(img, text_ids, text_attention_mask)
            loss = self.crit(logits, labels)

        # backward
        if self.use_amp:
            self.scaler.scale(loss).backward() # scale loss
            self.scaler.unscale_(self.optimizer) # unscale gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.get('max_grad_norm', 5))
            self.scaler.step(self.optimizer) # do a protected step update
            self.scaler.update() # update scaler
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.get('max_grad_norm', 5))
            self.optimizer.step()

        loss_val = loss.item()
        return loss_val, grad_norm

    def predict(self, batch):
        tensorized = unpack_batch(batch)
        labels = tensorized.pop('label')[0]

        img = tensorized['image'][0]
        text_ids, text_attention_mask = tensorized['text']

        # forward
        self.model.eval()
        logits = self.model(img, text_ids, text_attention_mask)
        loss = self.crit(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        return predictions, probs, loss.item()
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            exit()
        self.model.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.model.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
            'image_encoder': self.model.image_encoder.state_dict(),
            'text_encoder': self.model.text_encoder.state_dict(),
            'classifier': self.model.classifier.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            logger.info("model saved to {}".format(filename))
        except BaseException:
            logger.warning("[Warning: Saving failed... continuing anyway.]")

class NCEPretrainer():
    def __init__(self, opt):
        self.opt = opt

        self.model = JointNCEModel(self.opt)
        self.crit = nn.CrossEntropyLoss()
        self.model.cuda()
        self.crit.cuda()
        
        # separate bert optimization and the rest of the model if needed
        lr, bert_lr = opt['lr'], opt['bert_lr']
        wd, bert_wd = opt['weight_decay'], opt['bert_weight_decay']
        if lr == bert_lr and wd == bert_wd:
            # use unified lr and wd
            logger.info(f"Using same lr={lr} and wd={wd} for image and text encoders.")
            param_groups = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # use different lr and wd for different modules
            logger.info(f"Using separate learning parameters for image and text encoders.")
            logger.info(f"Image: lr={lr}, wd={wd}; Text: bert_lr={bert_lr}, bert_wd={bert_wd}")
            bert_params, other_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad: # skip non-gradient params
                    continue
                if 'text_encoder' in n:
                    bert_params.append(p)
                else:
                    other_params.append(p)
            param_groups = [ # make BERT an individual parameter group
                {'params': bert_params, 'lr': bert_lr, 'weight_decay': bert_wd},
                {'params': other_params}
            ]
        
        self.optimizer = get_optimizer(
            self.opt['optim'],
            param_groups,
            lr = lr,
            weight_decay=wd
        )
        
        # amp training
        self.use_amp = self.opt['amp']
        if self.use_amp:
            self.scaler = GradScaler()
        
        self.biloss = self.opt.get('biloss', False)
        self._lambda = self.opt.get('lambda', 0.5)
    
    def update(self, batch):
        tensorized = unpack_batch(batch)

        img = tensorized['image'][0]
        text_ids, text_attention_mask = tensorized['text']

        # make labels for NCE
        batch_size = img.size(0)
        labels = torch.cuda.LongTensor(list(range(batch_size)))

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            logits, logits2 = self.model(img, text_ids, text_attention_mask)
            loss = self.crit(logits, labels)
            if self.biloss:
                loss = self._lambda*loss + (1-self._lambda)*self.crit(logits2, labels) # add second text-to-image loss, labels are the same

        # backward
        if self.use_amp:
            self.scaler.scale(loss).backward() # scale loss
            self.scaler.unscale_(self.optimizer) # unscale gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.get('max_grad_norm', 5))
            self.scaler.step(self.optimizer) # do a protected step update
            self.scaler.update() # update scaler
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.get('max_grad_norm', 5))
            self.optimizer.step()
        loss_val = loss.item()
        return loss_val, grad_norm

    def predict(self, batch):
        tensorized = unpack_batch(batch)

        img = tensorized['image'][0]
        text_ids, text_attention_mask = tensorized['text']

        # make labels for NCE
        batch_size = img.size(0)
        labels = torch.cuda.LongTensor(list(range(batch_size)))

        # forward
        self.model.eval()
        logits, logits2 = self.model(img, text_ids, text_attention_mask)
        loss = self.crit(logits, labels)
        if self.biloss:
            loss = self._lambda*loss + (1-self._lambda)*self.crit(logits2, labels) # add second text-to-image loss
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        return predictions, labels.cpu().numpy().tolist(), loss.item()
    
    def get_metrics(self, y_true, y_pred, loss):
        assert len(y_true) == len(y_pred), "Must have same number of labels and predictions."
        assert len(y_true) > 0, "Labels cannot be empty."
        correct = len([t for t, p in zip(y_true, y_pred) if t == p])
        accu = correct / len(y_true) * 100
        metrics = {'accuracy': accu, 'loss': loss}
        return metrics
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            exit()
        self.model.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.model.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.model.image_proj.load_state_dict(checkpoint['image_proj'])
        self.model.text_proj.load_state_dict(checkpoint['text_proj'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
            'image_encoder': self.model.image_encoder.state_dict(),
            'text_encoder': self.model.text_encoder.state_dict(),
            'image_proj': self.model.image_proj.state_dict(),
            'text_proj': self.model.text_proj.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            logger.info("model saved to {}".format(filename))
        except BaseException:
            logger.warning("[Warning: Saving failed... continuing anyway.]")


def get_optimizer(optim, param_groups, lr, weight_decay, beta1=0.9, beta2=0.999, eps=1e-6, momentum=0):
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    elif optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))