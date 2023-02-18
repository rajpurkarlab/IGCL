import os
import argparse
import logging
import random
import time
import datetime
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.pretrain_loader import BinaryPretrainDataLoader, PretrainDataset, RIHPretrainDataset
from model.pretrainer import BinaryPretrainer
from utils.tracker import ProgressTracker
from utils import logging_config, disk
from utils.helper import ensure_dir, save_config, print_config, set_all_seeds

logger = logging.getLogger('transfer')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help="Data directory with train and valid indexed report files.")
    parser.add_argument('--meta_file', type=str, default='dataset/mimic-cxr/meta.json', help="Dataset meta file.")
    parser.add_argument('--img_dir', type=str, default='dataset/mimic-cxr/files/', help="Directory to load image data from.")
    parser.add_argument('--local_img_dir', type=str, default='zyh/mimic-cxr/files/', help="Directory to load image data from.")
    parser.add_argument('--image_encoder', type=str, default='resnet50', help="Name of the model architecture.")
    parser.add_argument('--bert_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help="Name of the pretrained BERT model.")
    parser.add_argument('--imsize', type=int, default=224, help="Size of image.")
    parser.add_argument('--augment_p', type=float, default=0.8, help="Probability for image augmentation.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate.")
    parser.add_argument('--finetune_text_encoder', dest='freeze_text_encoder', action='store_false', help="Whether to finetune text encoder.")

    parser.add_argument('--num_clf_layer', type=int, default=2, help="Number of layers to use for NN classifier.")
    parser.add_argument('--clf_hidden_dim', type=int, default=512, help="Number of hidden dims for NN classifier.")
    parser.add_argument('--pool', choices=['cls', 'mean', 'max'], default='mean', help="Type of pooling to use for text encoder.")

    parser.add_argument('--fp', dest='amp', action='store_false', help="Use full precision training; by default use mixed precision.")
    parser.add_argument('--rih', action='store_true', help="Train on the RIH data; use corresponding data loaders.")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam, adamw or adamax.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--annealing_factor', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the trained model; if None will use id to look up")
    parser.add_argument('--root_dir', type=str, default='saved_models/pretrain', help="Root directory for model saving.")
    parser.add_argument('--id', type=int, default=0, help="An id of the training run")
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    opt = vars(args)

    # seeds and cuda configs
    set_all_seeds(opt['seed'])
    torch.backends.cudnn.benchmark = True

    logger.info(f"Start pretraining with image encoder {opt['image_encoder']}...")

    if opt['save_dir'] is None:
        opt['save_dir'] = os.path.join(opt['root_dir'], str(opt['id']))
    ensure_dir(opt['save_dir'])
    logger.info(f"Saving model to {opt['save_dir']}")

    # save and print configs
    save_config(opt, os.path.join(opt['save_dir'], 'config.json'), verbose=False)
    print_config(opt)

    # try to find image directory on the local disk
    img_dir = disk.get_local_or_remote_dir(opt['img_dir'], opt['local_img_dir'], logger)

    # tokenizer and data loading
    logger.info(f"Loading data with data file from {opt['data_dir']}...")
    tokenizer = AutoTokenizer.from_pretrained(opt['bert_name'])
    train_file = os.path.join(opt['data_dir'], 'train.json')
    if opt['rih']:
        dataset_cls = RIHPretrainDataset
    else:
        dataset_cls = PretrainDataset
    
    train_set = dataset_cls(
        indexed_file=train_file,
        meta_file=opt['meta_file'],
        img_dir=img_dir,
        opt=opt,
        tokenizer=tokenizer,
        evaluation=False,
        imsize=opt['imsize'],
        augment_p=opt['augment_p']
    )
    train_loader = BinaryPretrainDataLoader(
        dataset=train_set,
        opt=opt,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['num_workers'],
        pin_memory=opt['pin_memory']
    )

    # model
    logger.info(f"Initializing model with image encoder {opt['image_encoder']}...")
    trainer = BinaryPretrainer(opt)
    progress_tracker = ProgressTracker(summary_dir=opt['save_dir'])

    # use loss as a criteria
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = trainer.optimizer,
        mode='min',
        factor=opt['annealing_factor'],
        patience=opt['patience'],
        verbose=True,
        min_lr=1e-8
    )

    # start pretraining
    step, global_step, current_epoch = 0, 0, 0
    best_avg_loss = 999.0
    should_stop = False
    start_time = datetime.datetime.now()
    log_start_time = time.time()
    logger.info(f"Start pretraining for {opt['num_epoch']} epochs and {opt['steps_per_epoch']} steps per epoch...")

    while True:
        for i, data in enumerate(train_loader):
            step += 1
            global_step += 1
            loss, grad_norm = trainer.update(data)
            progress_tracker.update_loss(loss, global_step, epoch=current_epoch+1)

            if step % opt['log_interval'] == 0:
                avg_loss = progress_tracker.get_avg_loss()
                progress_tracker.write_loss(avg_loss, global_step) # write summary
                duration_per_batch = (time.time() - log_start_time) / opt['log_interval']
                logger.info("| epoch {:3d} | {:5d}/{:5d} batches | sec/batch {:.6f} | loss {:5.4f} | grad norm {:3.4f}".format(
                    current_epoch + 1,
                    step,
                    opt['steps_per_epoch'],
                    duration_per_batch,
                    avg_loss,
                    grad_norm
                ))
                log_start_time = time.time() # update log start time
            
            if step % opt['steps_per_epoch'] == 0:
                current_epoch += 1
                step = 0
                epoch_avg_loss = progress_tracker.get_epoch_loss(current_epoch)
                progress_tracker.update_metrics({'avg_loss': epoch_avg_loss}, current_epoch)

                logger.info("| {:3d}/{:3d} epochs finished. | average loss {:.4f}".format(
                    current_epoch,
                    opt['num_epoch'],
                    epoch_avg_loss
                ))

                # update scheduler
                scheduler.step(epoch_avg_loss)
                # check if save model and update best score
                if epoch_avg_loss <= best_avg_loss:
                    # save model
                    trainer.save(os.path.join(opt['save_dir'], 'best_model.pt'))
                    best_avg_loss = epoch_avg_loss
                
                # recover training mode
                if current_epoch >= opt['num_epoch']:
                    should_stop = True
                    break
                
                log_start_time = time.time() # update log start time
        
        # end training
        if should_stop:
            break
    logger.info(f"Training ended with {current_epoch} epochs. Took {datetime.datetime.now() - start_time}\n")

if __name__ == "__main__":
    main()
