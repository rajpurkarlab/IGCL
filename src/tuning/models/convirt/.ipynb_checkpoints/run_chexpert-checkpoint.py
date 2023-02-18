"""
Train a model for the chexpert task.
"""
import os
import argparse
from tqdm import tqdm
import datetime
import time
import logging
import random
import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data.loader import CheXpertDataset, CHEXPERT_VALID_COLUMN_NAMES
from model import vision
from model.trainer import ImageTrainer
from utils.tracker import ProgressTracker
from utils.helper import ensure_dir, save_config, print_config, set_all_seeds
from utils import logging_config, disk

logger = logging.getLogger('transfer')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/chexpert', help="Directory to load image data from.")
    parser.add_argument('--local_img_dir', type=str, default='zyh/chexpert/', help="Directory to load image data from.")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help="Run training or eval.")
    parser.add_argument('--model_name', type=str, default='resnet50', help="Name of the model architecture.")
    parser.add_argument('--random_init', action='store_true', help="Use random init of the image model.")
    parser.add_argument('--pretrained', type=str, default='', help="Path to pretrained CNN model weights")
    parser.add_argument('--pretrained_key', type=str, default='image_encoder', help="A key that maps to the CNN model weights in the checkpoint file.")
    parser.add_argument('--imsize', type=int, default=224, help="Size of image.")
    parser.add_argument('--augment_p', type=float, default=0, help="Probability for image augmentation.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate.")
    parser.add_argument('--ratio', type=float, default=1.0, help="Ratio of training examples to use; default to use all")
    parser.add_argument('--clf_only', action='store_true', help="Freeze all weights but the classifier layer of the image model.")
    parser.add_argument('--test_file', type=str, default='test.csv', help="CSV file used in eval mode.")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--clf_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clf_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--stop_patience', type=int, default=5)
    parser.add_argument('--annealing_factor', type=float, default=0.5)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the trained model; if None will use id to look up")
    parser.add_argument('--root_dir', type=str, default='saved_models/chexpert', help="Root directory for model saving.")
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

    if opt['mode'] == 'train':
        train(opt)
    elif opt['mode'] == 'eval':
        eval(opt)
    else:
        raise Exception(f"Unsupported mode: {opt['mode']}")

def train(opt):
    logger.info("Training the CheXpert model...")

    if opt['save_dir'] is None:
        opt['save_dir'] = os.path.join(opt['root_dir'], str(opt['id']))
    ensure_dir(opt['save_dir'])
    logger.info("Saving model to {}".format(opt['save_dir']))

    # save and print options
    save_config(opt, os.path.join(opt['save_dir'], 'config.json'), verbose=False)
    print_config(opt)

    # try to find image directory on the local disk
    img_dir = disk.get_local_or_remote_dir(opt['data_dir'], opt['local_img_dir'], logger)

    # data loaders
    logger.info("Setting up data loaders...")
    loader_params = {
        'batch_size': opt['batch_size'],
        'num_workers': opt['num_workers'],
        'pin_memory': opt['pin_memory']
    }
    train_csv = os.path.join(opt['data_dir'], 'train.csv')
    val_csv = os.path.join(opt['data_dir'], 'valid.csv')
    opt['num_class'] = 2
    opt['num_label'] = len(CHEXPERT_VALID_COLUMN_NAMES)

    train_set = CheXpertDataset(
        img_dir=img_dir,
        csv_file=train_csv,
        imsize=opt['imsize'],
        evaluation=False,
        augment_p=opt['augment_p'],
        ratio=opt['ratio']
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_params)

    val_set = CheXpertDataset(
        img_dir=img_dir,
        csv_file=val_csv,
        imsize=opt['imsize'],
        evaluation=True
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_params)
    logger.info(f"Num of examples: training = {len(train_set)}, valid = {len(val_set)}")

    # trainer and model
    logger.info(f"Loading model {opt['model_name']} with pretrained weights...")
    trainer = ImageTrainer(opt, multilabel=True, random_init=opt['random_init'])
    # update model weights with specified pretrained model
    if len(opt['pretrained']) > 0:
        vision.reload_from_pretrained(trainer.model, filename=opt['pretrained'], key=opt['pretrained_key'], strict=False)
    
    if opt['clf_only']:
        trainer.freeze_all_but_clf()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=opt['annealing_factor'],
        patience=opt['patience'],
        verbose=True
    )
    progress_tracker = ProgressTracker(opt['save_dir'])

    # first train clf layer; skip if crf_only == True
    if not opt['clf_only'] and opt['clf_steps'] > 0:
        logger.info(f"Training clf for {opt['clf_steps']} steps...")
        step = 0
        stop = False
        start_time = time.time()
        while True:
            for i, data in enumerate(train_loader):
                _ = trainer.update(data, optimizer=trainer.clf_optimizer)
                step += 1
                if step % opt['log_interval'] == 0:
                    logger.info("| step {:5d}/{:5d} | duration {:5.2f} mins".format(
                        step,
                        opt['clf_steps'],
                        (time.time() - start_time) / 60
                    ))
                if step >= opt['clf_steps']:
                    stop = True
                    break
            if stop:
                break
        logger.info("Training clf finished.")

    # train full model
    steps_per_epoch = int(np.ceil(len(train_set) / opt['batch_size']))
    logger.info(f"Start training for {opt['max_epochs']} epochs, with {steps_per_epoch} steps per epoch...")
    # set true eval interval; for small training eval after every epoch
    if steps_per_epoch > opt['eval_interval']:
        eval_interval = opt['eval_interval']
    else:
        eval_interval = steps_per_epoch
    logger.info(f"Evaluate every {eval_interval} steps...")
    best_val_score = 0
    stopping = 0
    step, global_step, current_epoch = 0, 0, 1
    should_stop = False
    start_time = datetime.datetime.now()
    log_start_time = time.time()
    while True:
        step = 0
        logger.info("Starting epoch {:3d}/{:3d}...".format(current_epoch, opt['max_epochs']))
        for i, data in enumerate(train_loader):
            step += 1
            global_step += 1
            loss = trainer.update(data)
            progress_tracker.update_loss(loss, global_step)

            # logging
            if global_step % opt['log_interval'] == 0:
                avg_loss = progress_tracker.get_avg_loss()
                progress_tracker.write_loss(avg_loss, global_step) # write summary
                duration_per_batch = (time.time() - log_start_time) / opt['log_interval']
                logger.info("| epoch {:3d} | {:6d}/{:6d} batches | sec/batch {:.6f} | loss {:5.4f}".format(
                    current_epoch,
                    step,
                    steps_per_epoch,
                    duration_per_batch,
                    avg_loss
                ))
                log_start_time = time.time() # update log start time
            
            # validation
            if global_step % eval_interval == 0:
                logger.info('Running validation...')
                with torch.no_grad():
                    val_loss = 0
                    y_pred = []
                    for data in val_loader:
                        probs, loss = trainer.predict(data)
                        val_loss += loss
                        y_pred += probs
                y_pred = np.array(y_pred)
                y_true = np.array(val_set.labels)
                val_loss /= float(len(val_set)) # normalize loss
                val_metrics = trainer.get_metrics(y_true, y_pred, label_names=CHEXPERT_VALID_COLUMN_NAMES)
                val_score = val_metrics['auc']
                progress_tracker.update_metrics(val_metrics, global_step)

                logger.info("| validation @ {:3d} epochs {} steps | loss {:.4f} | avg f1 {:.4f} | avg auc {:.4f}".format(
                    current_epoch,
                    step,
                    val_loss,
                    val_metrics['f1'],
                    val_metrics['auc']
                ))
                logger.info("\n" + trainer.pretty_print_metrics(val_metrics, CHEXPERT_VALID_COLUMN_NAMES))

                # update scheduler
                scheduler.step(val_score)
                # check if save model and update best score
                if val_score >= best_val_score:
                    # save model
                    trainer.save(os.path.join(opt['save_dir'], 'best_model.pt'))
                    best_val_score = val_score
                    stopping = 0
                else:
                    stopping += 1
                # do early stop when patience is reached
                if stopping >= opt['stop_patience']:
                    should_stop = True
                
                log_start_time = time.time() # update log start time
        
        current_epoch += 1
        if current_epoch >= opt['max_epochs']:
            should_stop = True
        
        # end training
        if should_stop:
            break
    logger.info('Training ended with {} epochs. Took {}\n'.format(current_epoch, datetime.datetime.now() - start_time))
    return

def eval(opt):
    logger.info("Evaluating CheXpert model...")

    # load opt and states from file
    if opt['save_dir'] is None:
        opt['save_dir'] = os.path.join(opt['root_dir'], str(opt['id']))
    model_file = os.path.join(opt['save_dir'], 'best_model.pt')
    logger.info(f"Loading model states from {model_file}...")
    if not os.path.exists(model_file):
        raise Exception(f"Model file not found at: {model_file}")
    model_state, loaded_opt = ImageTrainer.load_from_file(model_file)
    assert loaded_opt['num_label'] == len(CHEXPERT_VALID_COLUMN_NAMES), \
        "Number of labels in loaded model does not match the label number for CheXpert."

    # build model and load weights
    trainer = ImageTrainer(loaded_opt, multilabel=True)
    trainer.model.load_state_dict(model_state)

    # try to find image directory on the local disk
    img_dir = disk.get_local_or_remote_dir(opt['data_dir'], opt['local_img_dir'], logger)

    # data loaders
    logger.info(f"Loading data from {opt['test_file']}...")
    test_csv = os.path.join(opt['data_dir'], opt['test_file'])
    test_set = CheXpertDataset(
        img_dir=img_dir,
        csv_file=test_csv,
        imsize=loaded_opt['imsize'],
        evaluation=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=opt['num_workers'],
        pin_memory=opt['pin_memory']
    )
    logger.info(f"Num of examples: test = {len(test_set)}")

    logger.info('Start running evaluation...')
    with torch.no_grad():
        y_pred = []
        for batch in tqdm(test_loader):
            probs, _ = trainer.predict(batch)
            y_pred += probs
    y_pred = np.array(y_pred)
    y_true = np.array(test_set.labels)

    metrics = trainer.get_metrics(y_true, y_pred, label_names=CHEXPERT_VALID_COLUMN_NAMES)
    out_str = trainer.pretty_print_auc_from_metrics(metrics)
    logger.info("Test AUC scores:")
    logger.info("\n" + out_str)

    auc_file = os.path.join(opt['save_dir'], 'auc.txt')
    with open(auc_file, 'w') as outfile:
        print(out_str, file=outfile)

if __name__ == '__main__':
    main()
