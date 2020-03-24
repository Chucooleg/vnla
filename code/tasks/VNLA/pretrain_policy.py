import os
import sys
import numpy as np
import torch
from env import VNLAPretrainBatch
from policy_pretrainer import PolicyPretrainer
from tensorboardX import SummaryWriter
from model import SwapClassifier
from eval import SwapEvaluation, compute_auc

def set_path():

    # Set general output dir
    hparams.exp_dir = os.getenv('PT_EXP_DIR')
    hparams.model_prefix = '%s' % (hparams.exp_name)

    # Set tensorboard log dir
    if hparams.local_run:
        hparams.tensorboard_dir = hparams.exp_dir
    else:
        if hparams.plot_to_philly:
            hparams.tensorboard_dir = os.environ.get('PHILLY_LOG_DIRECTORY', '.')
        else:
            hparams.tensorboard_dir = os.environ.get('PT_TENSORBOARD_DIR', '.')
    print ("tensorboard dir = {}".format(hparams.tensorboard_dir))   

    # Set pretrain data load path
    DATA_DIR = os.getenv('PT_DATA_DIR')
    # hparams.data_dir should be 'pretrain'
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)  # $PT_DATA_DIR/pretrain/
    hparams.img_features = os.path.join(DATA_DIR, hparams.img_features)

    # where to checkpoint
    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.model_prefix) 
    
def load(path):
    ckpt = torch.load(path, map_location=self.device)
    hparams = ckpt['hparams']

    # overwrite hparams by args
    for flags in var(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()
    return ckpt

def save(path, model, optimizer, iter, best_metrics, train_env):
    ckpt = {
        'mode_state_dict':  model.state_dict()
        'optim_state_dict': optimizer.state_dict()
        'hparams':          hparams,
        'iter':             iter,
        'best_metrics':     best_metrics,
        'data_idx':         train_env.ix,
    }
    torch.save(ckpt, path)

def setup(seed=None):

    if seed is not None:
        hparams.seed = seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(train_env, val_envs, trainer, model, optimizer, start_iter, end_iter, best_metrics, eval_mode):

    if not eval_mode:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)

    for idx in range(start_iter, end_iter, hparams.log_every):
        # An iter is a batch

        interval = min(hparams.log_every, end_iter - idx) # An interval is a number of batches
        iter = idx + interval

        # Train for log_every iteration
        if eval_mode:
            loss_str = '\n * eval mode'
        else:
            trainer.train(train_env, optimizer, interval, idx)

            train_losses = np.array(trainer.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)

            loss_str = '\n * train loss: %.4f' % train_loss_avg

            SW.add_scalar('train loss per {} iters'.format(hparams.log_every), train_loss_avg, iter)

            metrics = defaultdict(dict)
            should_save_ckpt = []

        metrics = defaultdict(dict)
        res = defaultdict(dict)

        # Run valiation
        for env_name, (env, evaluator) in val_envs.items():
            trainer.test(env, idx)

            val_loss_avg = np.average(trainer.losses)
            loss_str += '\n * %s loss: %.4f' % (env_name, val_loss_avg)

            SW.add_scalar('{} loss per {} iters'.format(env_name, hparams.log_every), val_loss_avg, iter)

            trainer.results_path = os.path.join(hparams.exp_dir, '%s_%s_for_eval.json' % (hparams.model_prefix, env_name))
            trainer.write_results()
            auc, accuracy, num_pts, preds, tars = evaluator.score(trainer.results_path)

            SW.add_scalar('{} accuracy per {} iters'.format(env_name, hparams.log_every ), accuracy, iter)

            metrics['auc'][env_name] = auc
            metrics['accuracy'][env_name] = (accuracy, num_pts)

            res['preds'][env_name] = preds
            res['tars'][env_name] = tars

            if not eval_mode and metrics['auc'][env_name] > best_metrics[env_name]:
                should_save_ckpt.append(env_name)
                best_metrics[env_name] = metrics['auc'][env_name]
                print('best %s success rate %.3f' % (env_name, best_metrics[env_name]))

        if not eval_mode:
            combined_metric = compute_auc(res)
            assert combined_metric == combined_metric and combined_metric > 0.0
            if combined_metric > best_metrics['combined']:
                best_metrics['combined'] = combined_metric
                print ('best combined accuracy %.3f' % combined_metric)

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
            iter, float(iter)/end_iter*100, loss_str))

        if not eval_mode:
            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            # Save lastest model?
            if iter == end_iter or iter % hparams.save_every == 0:
                should_save_ckpt.append('last')

            for env_name in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.model_prefix, env_name))
                save(save_path, model, optimizer, iter, best_metrics, train_env)
                print("Saved %s model to %s" % (env_name, save_path))

        # progress per training job on philly
        print("PROGRESS: {}%".format(float(iter)/end_iter*100))

    return None  

def train_val(device, seed=None):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    # Resume from lastest checkpoint (if any)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path, device)
        start_iter = ckpt['iter']
    else:
        if hasattr(args, 'load_path') and hasattr(args, 'eval_only') and args.eval_only:
            sys.exit('load_path %s does not exist!' % hparams.load_path)
        ckpt = None
        start_iter = 0
    end_iter = hparams.n_iters

    # Setup seed and read vocab
    setup(seed=seed)

    # Create training environment
    train_env = VNLAPretrainBatch(hparams, split='train')

    # Create validation environments
    val_splits = ['val_seen', 'val_unseen']
    eval_mode = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_mode:
        if '_unseen' in hparams.load_path:
            val_splits = ['val_unseen']
        if '_seen' in hparams.load_path:
            val_splits = ['val_seen']
        end_iter = start_iter + hparams.log_every

    val_envs = { split: (VNLAPretrainBatch(hparams, split=split, tokenizer=tok,
        from_train_env=train_env),
        SwapEvaluation(hparams, [split], hparams.data_path)) for split in val_splits}

    # Build model
    model = SwapClassifier(hparams, device)

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    print ("device count :".format(torch.cuda.device_count()))

    best_metrics = { 'val_seen'  : -1,
                     'val_unseen': -1,
                     'combined'  : -1 }

    # Load model parameters from a checkpoint (if any)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        best_metrics = ckpt['best_metrics']
        train_env.ix = ckpt['data_idx']

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)

    # Initializa trainer
    trainer = PolicyPretrainer(model, hparams, device)

    # Train
    return train(train_env, val_envs, trainer, model, optimizer, start_iter, end_iter, best_metrics, eval_mode)


if __name__ == '__main__':

    os.environ['PT_DATA_DIR'] = '/home/hoyeung/blob_matterport3d/'

    # load hparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file', type=str,
        help='configuration file')
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()
    print ("set_path() is done")

    device = torch.device('cuda')

    # Train / Val !
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

    # Train / Val !
    train_val(device)

# ------------------------------------------------------------------------------