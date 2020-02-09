import os
import time
import sys
import multiprocessing
import random
import numpy as np

from pprint import pprint
import argparse
from argparse import Namespace
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import FFSemanticClassifier
from dataset import PanoramaDataset
import utils
from utils import timeSince
from flags_semantics import make_parser

def set_path():
    '''set paths for data, saving, logging'''

    # Set general output dir
    hparams.exp_dir = os.getenv('PT_EXP_DIR')

    # Set model prefix
    assert hparams.layers >= 1
    hparams.model_prefix = "semantic_classifier_layers_{}_lr_{}".format(hparams.layers, hparams.lr)

    # Set tensorboard log dir
    if hparams.local_run:
        hparams.tensorboard_dir = hparams.exp_dir
    else:
        if hparams.plot_to_philly:
            hparams.tensorboard_dir = os.environ.get('PHILLY_LOG_DIRECTORY', '.')
        else:
            hparams.tensorboard_dir = os.environ.get('PT_TENSORBOARD_DIR', '.')
    print ("tensorboard dir = {}".format(hparams.tensorboard_dir))        

    # Set model load path
    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
            os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.model_prefix)

    # Set data load path
    DATA_DIR = os.getenv('PT_DATA_DIR')
    hparams.data_path = DATA_DIR
    hparams.img_features = os.path.join(DATA_DIR, hparams.img_features)

    # Set scan path
    hparams.scans_path = os.path.join(DATA_DIR, hparams.scans_path)
    hparams.room_types_path = os.path.join(DATA_DIR, hparams.room_types_path)

    # Where to save training and validation indices
    hparams.tr_idx_save_path = os.path.join(DATA_DIR, hparams.tr_idx_save_path) if hasattr(hparams, 'tr_idx_save_path') else \
        os.path.join(hparams.exp_dir, 'train_indices.txt')
    hparams.val_seen_idx_save_path = os.path.join(DATA_DIR, hparams.val_seen_idx_save_path) if hasattr(hparams, 'val_seen_idx_save_path') else \
        os.path.join(hparams.exp_dir, 'val_seen_indices.txt')
    hparams.val_unseen_idx_save_path = os.path.join(DATA_DIR, hparams.val_unseen_idx_save_path) if hasattr(hparams, 'val_unseen_idx_save_path') else \
        os.path.join(hparams.exp_dir, 'val_unseen_indices.txt')

def setup(seed=None):
    '''
    Set up seeds.
    '''
    if seed is not None:
        hparams.seed = seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save(path, model, optimizer, epoch, best_metrics):
    '''save model checkpt'''

    ckpt = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'hparams'         : hparams,
            'epoch'           : epoch,
            'best_metrics'    : best_metrics,
        }
    torch.save(ckpt, path)

def load(path, device):
    global hparams
    ckpt = torch.load(path, map_location=device)
    hparams = ckpt['hparams']

    # Overwrite hparams by args
    for flg in vars(args):
        val = getattr(args, flg)
        if val is not None:
            setattr(hparams, flg, val)

    set_path()
    return ckpt

def retrieve_rm_labels_and_feature_ids(scans):
    rm_labels = []
    feature_ids = []
    for scan in scans:
        scan_G = utils.load_nav_graphs(scan)
        scan_panos_to_region = utils.load_panos_to_region(scan,"")
        for n in scan_G.nodes:
            room_label_str = scan_panos_to_region[n]
            long_id = scan + '_' + n
            for viewix in range(36):
                rm_labels.append(room_label_str)
                feature_ids.append((long_id, viewix))
    return rm_labels, feature_ids

def read_rm_labels_and_feature_ids(idx_save_path, room_types, image_extent):
    feature_ids = []
    room_labels = []
    with open(idx_save_path, 'r') as fh:
        lines = fh.read().split('\n')
        lines = lines[:-1]
        for line in lines:
            long_id, viewix, room_label = line.split('\t')
            if image_extent != 'single':
                room_labels.append(room_types.index(room_label))
                feature_ids.append((long_id, eval(viewix)))
            else:
                room_labels.append(room_types.index(room_label))
                feature_ids.append((long_id, int(viewix)))
    return feature_ids, room_labels

def train_val(device, seed=None):
    ''' 
    Train on the training set, and validate on seen and unseen splits.
    - load last model 
    - define Datasets for training and validation.
    - specify choice of image classifier model

    Returns:
        - calls main train() i.e. the main training loop
    '''

    # Resume from latest checkpoint (if any)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path, device)
        start_epoch = ckpt['epoch']
    else:
        if hasattr(args, 'load_path') and hasattr(args, 'eval_only') and args.eval_only:
            sys.exit('load_path %s does not exist!' % hparams.load_path)
        ckpt = None
        start_epoch = 0
    end_epoch = hparams.n_epochs

    # Setup seed and read vocab
    setup(seed=seed)

    # Set eval iters 
    eval_only = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_only:
        end_epoch = 1

    # Define scans
    with open(hparams.scans_path, "r") as fh:
        scans = fh.read().split("\n")
        scans = scans[:-1]

    # Define room types
    with open(hparams.room_types_path, "r") as fh:
        room_types = fh.read().split("\n")
        room_types = room_types[:-1]

    # Load image features
    # image_h, image_w, vfov, all_img_features = utils.load_img_features(hparams.img_features)
    image_h, image_w, vfov, all_img_features = utils.load_img_features_as_tensors(hparams.img_features)

    # ---------- Split scans and create Datasets
    if os.path.exists(hparams.tr_idx_save_path) and os.path.exists(hparams.val_seen_idx_save_path) and \
        os.path.exists(hparams.val_unseen_idx_save_path):
        
        print ("index and label save path exists, loading from these text files...")
        print ("{}\n{}\n{}".format(hparams.tr_idx_save_path, hparams.val_seen_idx_save_path, hparams.val_unseen_idx_save_path))
        tr_seen_feature_ids, tr_seen_rm_labels = read_rm_labels_and_feature_ids(hparams.tr_idx_save_path, room_types, hparams.image_extent)
        val_seen_feature_ids, val_seen_rm_labels = read_rm_labels_and_feature_ids(hparams.val_seen_idx_save_path, room_types, hparams.image_extent)
        val_unseen_feature_ids, val_unseen_rm_labels = read_rm_labels_and_feature_ids(hparams.val_unseen_idx_save_path, room_types, hparams.image_extent)

    elif hparams.image_extent == 'single':

        random.shuffle(scans)

        # Split into seen and unseen scans
        val_unseen_scans = np.random.choice(scans, size=hparams.n_unseen_scans, replace=False)
        # still includes both train seen and val seen
        seen_scans = [scan for scan in scans if scan not in val_unseen_scans]

        # Expand into datapts -- feature_ids and labels 
        seen_rm_labels, seen_feature_ids = retrieve_rm_labels_and_feature_ids(seen_scans)
        val_unseen_rm_labels, val_unseen_feature_ids = retrieve_rm_labels_and_feature_ids(val_unseen_scans)

        # Sample val seen idx from seen datapts
        val_seen_idx = np.random.choice(len(seen_feature_ids), size=len(val_unseen_feature_ids), replace=False)

        # Get tr and val seen datapts
        tr_seen_rm_labels = []
        tr_seen_feature_ids = []
        val_seen_rm_labels = []
        val_seen_feature_ids = []
        for i in range(len(seen_feature_ids)):
            if i in val_seen_idx:
                val_seen_rm_labels.append(seen_rm_labels[i])
                val_seen_feature_ids.append(seen_feature_ids[i])
            else:
                tr_seen_rm_labels.append(seen_rm_labels[i])
                tr_seen_feature_ids.append(seen_feature_ids[i])

        # Save a record of train, val seen and val unseen indices and labels on disk
        with open(hparams.tr_idx_save_path, "w") as fh:
            for i, feat_id in enumerate(tr_seen_feature_ids):
                fh.write("{}\t{}\t{}\n".format(feat_id[0], feat_id[1], tr_seen_rm_labels[i]))

        with open(hparams.val_seen_idx_save_path, "w") as fh:
            for i, feat_id in enumerate(val_seen_feature_ids):
                fh.write("{}\t{}\t{}\n".format(feat_id[0], feat_id[1], val_seen_rm_labels[i]))         

        with open(hparams.val_unseen_idx_save_path, "w") as fh:
            for i, feat_id in enumerate(val_unseen_feature_ids):
                fh.write("{}\t{}\t{}\n".format(feat_id[0], feat_id[1], val_unseen_rm_labels[i]))

        # convert room labels from string to integer indices
        tr_seen_rm_labels = [room_types.index(lab) for lab in tr_seen_rm_labels]
        val_seen_rm_labels = [room_types.index(lab) for lab in val_seen_rm_labels]
        val_unseen_rm_labels = [room_types.index(lab) for lab in val_unseen_rm_labels]

    else:
        raise ValueError('Check arguments for hparams.image_extent. If using more than one frame from panorama to make classification, must read from predefined indices & label files.')

    # Make torch datasets
    tr_dataset = PanoramaDataset(tr_seen_rm_labels, tr_seen_feature_ids, all_img_features)
    val_seen_dataset = PanoramaDataset(val_seen_rm_labels, val_seen_feature_ids, all_img_features)
    val_unseen_dataset = PanoramaDataset(val_unseen_rm_labels, val_unseen_feature_ids, all_img_features)      
    # ---------- 

    # Debug only
    assert len(room_types) == 30

    # Build model
    model = FFSemanticClassifier(hparams, room_types).to(device)

    # Specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    # Loss critierion
    criterion = nn.CrossEntropyLoss()

    # used to determine if we should checkpt the model
    best_metrics = { 'val_seen'  : -1,
                     'val_unseen': -1,
                     'combined'  : -1 }

    # Load model parameters from a checkpoint (if any)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        best_metrics = ckpt['best_metrics']

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)

    # Train and validate
    return train(tr_dataset, val_seen_dataset, val_unseen_dataset, room_types, model, optimizer, criterion, start_epoch, end_epoch, best_metrics, eval_only, device)

def train(tr_dataset, val_seen_dataset, val_unseen_dataset, room_types, model, optimizer, criterion, start_epoch, end_epoch, best_metrics, eval_only, device):
    '''
    Main training loop. Use cross-validation. For each train-val split combination, loop through hparams.n_epoch.
    - calls train() which includes backprop
    - calls test() which writes results to json files
    - print losses and metrics to stdout using `loss_str`
    '''
    start = time.time()

    SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)

    if not eval_only:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    # n_workers = max(1, multiprocessing.cpu_count() - 2) if not hparams.local_run else 2
    n_workers = 2
    tr_data_loader = DataLoader(tr_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=n_workers)
    val_seen_data_loader = DataLoader(val_seen_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=n_workers)
    val_unseen_data_loader = DataLoader(val_unseen_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=n_workers)
    print ("finish constructing dataloaders")

    # Initialize loss string to be printed to stdout
    loss_str = ''

    itr = 0
    for epoch in range(start_epoch, end_epoch + 1):

        if eval_only:
            loss_str = '\n * eval mode' 
        else:
            model.train()
            training_loss = []
            epoch_start_time = time.time()
            print ("epoch {} starts".format(epoch))
            for i, data in enumerate(tr_data_loader, 0):
                itr += 1
                # if itr % 1 == 0: 
                #     print ('itr = {}'.format(itr))
                # shape (batch size, 2048)
                features = data['feature'].to(device)
                # print ('itr = {}'.format(itr))
                # print ('itr = {}, time = {}'.format(itr, time.time() - iter_start_time))
                
                # shape (batch_size,)
                rooms = data['room'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()   
                # forward pass
                # shape (batch_size, 30)
                logits = model(features)
                # backprop
                loss = criterion(logits, rooms)
                loss.backward()
                optimizer.step()
                # accumulate loss
                loss_per_iter = loss.item()
                training_loss.append(loss_per_iter)
                SW.add_scalar('loss per iter', loss_per_iter, itr)
                # print ("iter time = {}".format(time.time() - iter_start_time))
                
            print ("epoch time = {}".format(time.time() - epoch_start_time))
            train_loss_avg = np.average(training_loss)
            loss_str = '\n * train loss: %.4f' % train_loss_avg
            SW.add_scalar('train loss per epoch', train_loss_avg, epoch)
        
        model.eval()
        metrics = defaultdict(dict)
        should_save_ckpt = []  # save if model is the best for seen, unseen
        with torch.no_grad():
            for split, dataloader, dataset_len in zip(['val_seen', 'val_unseen'], [val_seen_data_loader, val_unseen_data_loader], [len(val_seen_dataset), len(val_unseen_dataset)]):

                val_losses = []
                total = 0
                top1_corrects = 0
                top3_corrects = 0
                top5_corrects = 0
                results = []
                val_itr = 0
                for i, data in enumerate(dataloader, 0):
                    
                    val_itr += 1

                    # shape (batch_size,)
                    long_ids = data['long_id']
                    viewixs = data['viewix']

                    # shape (batch_size, 2048)
                    features = data['feature'].to(device)
                    # shape (batch_size,)
                    rooms = data['room'].to(device)
                    # shape (batch_size, 30)
                    logits = model(features)

                    # loss
                    loss = criterion(logits, rooms)
                    val_losses.append(loss.item())
                    # top1 accuracy
                    logits_data = logits.data
                    top1_pred = torch.max(logits_data, 1)[1]
                    top1_correct = (top1_pred == rooms).sum().item()
                    top1_corrects += top1_correct
                    # top3 accuracy
                    top2_pred = logits_data.topk(k=2, dim=1)[1][:,1]
                    top2_correct = top1_correct + (top2_pred == rooms).sum().item()
                    top3_pred = logits_data.topk(k=3, dim=1)[1][:,2]
                    top3_correct = top2_correct + (top3_pred == rooms).sum().item()
                    top3_corrects += top3_correct
                    # top5 accuracy
                    top4_pred = logits_data.topk(k=4, dim=1)[1][:,3]
                    top4_correct = top3_correct + (top4_pred == rooms).sum().item()
                    top5_pred = logits_data.topk(k=5, dim=1)[1][:,4]
                    top5_correct = top4_correct + (top5_pred == rooms).sum().item()
                    top5_corrects += top5_correct
                    # total
                    total += rooms.shape[0]

                    if epoch == end_epoch or (epoch > 0 and epoch % hparams.save_every == 0):
                        # write results
                        logits_list = logits_data.tolist()
                        rooms_list = rooms.data.tolist()
                        viewixs_list = viewixs.tolist()
                        
                        for j, feat_id in enumerate(zip(long_ids, viewixs_list)):
                            results.append({
                                'feature_id': feat_id,
                                'room_label': rooms_list[j],
                                'logits': logits_list[j]})
                
                top1_accuracy = top1_corrects * 1.0 / total
                top3_accuracy = top3_corrects * 1.0 / total
                top5_accuracy = top5_corrects * 1.0 / total
                val_loss_avg = np.average(val_losses)

                loss_str += '\n * %s loss: %.4f' % (split, val_loss_avg)
                loss_str += ', %s: %.4f' % ('top 1 accuracy', top1_accuracy)
                loss_str += ', %s: %.4f' % ('top 3 accuracy', top3_accuracy)
                loss_str += ', %s: %.4f' % ('top 5 accuracy', top5_accuracy)

                SW.add_scalar('{} loss per epoch'.format(split), val_loss_avg, epoch)
                SW.add_scalar('{} top 1 accuracy per epoch'.format(split), top1_accuracy, epoch)
                SW.add_scalar('{} top 3 accuracy per epoch'.format(split), top3_accuracy, epoch)
                SW.add_scalar('{} top 5 accuracy per epoch'.format(split), top5_accuracy, epoch)             
                metrics[split]['top1_correct'] = top1_correct
                metrics[split]['total'] = total

                if epoch == end_epoch or (epoch > 0 and epoch % hparams.save_every == 0):
                    # check if only unique ids are kept
                    assert len(results) == dataset_len
                    # Write results to files
                    with open(os.path.join(hparams.exp_dir, '{}_results.json'.format(split)), "w") as fh:
                        json.dump(results, fh)

                # if top1 accuracy in this interval is better than the last, save the checkpt
                if not eval_only and top1_accuracy > best_metrics[split]:
                    should_save_ckpt.append(split)
                    best_metrics[split] = top1_accuracy
                    print('%s best top1 accuracy at %.3f' % (split, top1_accuracy))

            combined_metric = (metrics['val_seen']['top1_correct'] + metrics['val_unseen']['top1_correct']) / (metrics['val_seen']['total'] + metrics['val_unseen']['total'])
                
            SW.add_scalar('combined top 1 accuracy per epoch', combined_metric, epoch)
            print("combined top 1 acc = {}".format(combined_metric))

            # save best combined model if we are not running eval
            if not eval_only:
                if combined_metric > best_metrics['combined']:
                    should_save_ckpt.append('combined')
                    best_metrics['combined'] = combined_metric

        # log time and loss to terminal
        print('%s (%d %d%%) %s' % (timeSince(start, float(epoch + 1)/end_epoch),
                epoch, float(epoch)/end_epoch*100, loss_str))

        # save checkpt
        if not eval_only:

            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and epoch >= hparams.start_lr_decay and epoch % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            # Save lastest model
            if epoch == end_epoch or epoch % hparams.save_every == 0:
                should_save_ckpt.append('last')

            for split in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.model_prefix, split))
                save(save_path, model, optimizer, epoch, best_metrics)
                print("Saved %s model to %s" % (split, save_path))        

            # log time again after saving
            print('%s (%d %d%%)' % (timeSince(start, float(epoch + 1)/end_epoch),
                    epoch, float(epoch)/end_epoch*100))        

        # progress per training job on philly
        print("PROGRESS: {}%".format(float(epoch)/end_epoch*100))

    return None

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()

    # Read config from json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()

    device = torch.device('cuda')
    print ("CUDA device count = {}".format(torch.cuda.device_count()))

    train_val(device)


# TURN OFF WHEN NOT USING VS code debugger------------------------------------------------------------------------------
hparams = None
args = None

def vs_code_debug(args_temp):
    global hparams
    global args

    parser = make_parser()
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args_temp["config_file"]) as f:
        hparams = Namespace(**json.load(f)) 

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)   

    # Overwrite hparams by args_temp
    for flag in args_temp:
        value = args_temp[flag]
        setattr(hparams, flag, value)

    set_path()
    print ("set_path() is done")

    device = torch.device('cuda')

    print ("CUDA device count = {}".format(torch.cuda.device_count()))

    train_val(device)
