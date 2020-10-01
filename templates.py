def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_siren'):
        args.mode = 'train'

        args.dataset_code = 'samples'

        args.save_models_to_disk = 'True'
        args.dataloader_code = 'samples'
        args.img_name = '100'

        batch = 10000
        seed = 25

        args.log_period_as_iter = 1

        args.dataloader_random_seed = seed
        args.dataloader_workers = 8
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed

        args.trainer_code = 'samples'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.00001
        args.lr = 0.0001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'cos'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'loss'

        args.model_code = 'sirenFC2D'
        args.model_init_seed = seed

        args.dropout = 0.1

        args.hparams_to_log = ['model_init_seed', 'weight_decay', 'lr', 'num_epochs', 'model_code']
        args.metrics_to_log = ['loss']
        args.experiment_dir = 'experiments/randomtests'
        args.experiment_description = 'del'