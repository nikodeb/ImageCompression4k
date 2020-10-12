from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def setup_model_args(args_in, lr):
    model_args = args_in
    args.mode = 'train'

    model_args.dataset_code = 'samples'

    model_args.save_models_to_disk = 'True'
    model_args.dataloader_code = 'samples'
    model_args.img_name = '1001'
    model_args.img_resize_height = 216
    model_args.img_resize_width = 318
    model_args.normalise_coords = 'negpos1'

    batch = 1
    seed = 25

    args.log_period_as_iter = 1

    model_args.dataloader_random_seed = seed
    model_args.dataloader_workers = 0
    model_args.load_full_img = 'T'
    model_args.train_batch_size = batch
    model_args.val_batch_size = batch
    model_args.test_batch_size = batch
    model_args.dataset_split_seed = seed

    model_args.trainer_code = 'samples'
    model_args.device = 'cuda'
    model_args.num_gpu = 1
    model_args.device_idx = '0'
    model_args.optimizer = 'Adam'
    model_args.weight_decay = 0
    model_args.lr = lr
    model_args.enable_lr_schedule = False
    model_args.lr_sched_type = 'cos'
    model_args.num_warmup_steps = 20
    model_args.decay_step = 25
    model_args.gamma = 1.0
    model_args.num_epochs = 500
    model_args.metric_ks = [1, 5, 10, 20, 50, 100]
    model_args.best_metric = 'loss'

    model_args.model_code = 'siren'
    model_args.model_init_seed = seed

    model_args.dropout = 0.1

    model_args.hparams_to_log = ['model_init_seed', 'weight_decay', 'lr', 'num_epochs', 'model_code']
    model_args.metrics_to_log = ['loss']
    model_args.experiment_dir = 'experiments/siren_64_64_0hid'
    model_args.experiment_description = '{}'.format(model_args.lr)
    return model_args

def train(args_in):
    export_root = setup_train(args_in)
    train_loader, val_loader, test_loader = dataloader_factory(args_in)
    model = model_factory(args_in)
    trainer = trainer_factory(args_in, model, train_loader, val_loader, test_loader, export_root)
    if args_in.mode == 'train':
        trainer.train()
    trainer.test()

if __name__ == '__main__':
    # lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    for lr in lrs:
        model_args = setup_model_args(args, lr=lr)
        if args.mode == 'train' or args.mode == 'test':
            train(model_args)
        else:
            raise ValueError('Invalid mode')
