# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from main_worker import main_worker

# from imaginaire.config import Config
# from imaginaire.utils.cudnn import init_cudnn
# from imaginaire.utils.dataset import get_train_and_val_dataloader
# from imaginaire.utils.distributed import init_dist
# from imaginaire.utils.distributed import master_only_print as print
# from imaginaire.utils.gpu_affinity import set_affinity
# from imaginaire.utils.logging import init_logging, make_logging_dir
# from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
#                                       get_trainer, set_random_seed)


# torch.backends.cudnn.benchmark = True


# def main_worker(gpu, hydra_cfg):
#     print(gpu)
#     set_affinity(gpu)
#     set_random_seed(hydra_cfg.seed, by_rank=True)
#     cfg = Config(hydra_cfg.config)
#     if hydra_cfg.patch_wise:
#         cfg.dis.patch_wise = True

#     if hydra_cfg.gpus_per_node > 1 or hydra_cfg.num_nodes > 1:
#         # For multiprocessing distributed training, rank needs to be the
#         # global rank among all the processes. Think this is not computed correctly here for multi
#         # node training.
#         local_rank = gpu
#         hydra_cfg.local_rank = local_rank
#         cfg.local_rank = local_rank
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '2245'
#         dist.init_process_group(backend='nccl', init_method='env://',
#                                 world_size=hydra_cfg.num_nodes * hydra_cfg.gpus_per_node, rank=local_rank)    
#         torch.cuda.set_device(gpu)

#     print('made it past init process')
#     # Override the number of data loading workers if necessary
#     if hydra_cfg.num_workers is not None:
#         cfg.data.num_workers = hydra_cfg.num_workers

#     # Create log directory for storing training results.
#     # if hydra_cfg.local_rank == 0:
#     cfg.date_uid, cfg.logdir = init_logging(hydra_cfg.config, hydra_cfg.logdir)
#     make_logging_dir(cfg.logdir)

#     # TODO(eugenevinitsky) put this on the right device!!!
#     # Initialize data loaders and models.
#     train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg)
#     net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
#         get_model_optimizer_and_scheduler(cfg, seed=hydra_cfg.seed, device=gpu)
#     trainer = get_trainer(cfg, net_G, net_D,
#                           opt_G, opt_D,
#                           sch_G, sch_D,
#                           train_data_loader, val_data_loader, gpu)
#     # TODO(make the loading work more carefully using map location)
#     current_epoch, current_iteration = trainer.load_checkpoint(
#         cfg, hydra_cfg.checkpoint)

#     # Start training.
#     for epoch in range(current_epoch, cfg.max_epoch):
#         print('Epoch {} ...'.format(epoch))
#         if hydra_cfg.gpus_per_node > 1 or hydra_cfg.num_nodes > 1:
#             train_data_loader.sampler.set_epoch(current_epoch)
#         trainer.start_of_epoch(current_epoch)
#         for it, data in enumerate(train_data_loader):
#             data = trainer.start_of_iteration(data, current_iteration)
#             for key, val in data.items():
#                 if isinstance(val, torch.Tensor):
#                     data[key] = val.to(gpu)

#             for _ in range(cfg.trainer.dis_step):
#                 trainer.dis_update(data)
#             for _ in range(cfg.trainer.gen_step):
#                 trainer.gen_update(data)

#             current_iteration += 1
#             trainer.end_of_iteration(data, current_epoch, current_iteration)
#             if current_iteration >= cfg.max_iter:
#                 print('Done with training!!!')
#                 return

#         current_epoch += 1
#         trainer.end_of_epoch(data, current_epoch, current_iteration)
#     print('Done with training!!!')
#     return

@hydra.main(config_path='.', config_name='config')
def main(cfg):
    if cfg.gpus_per_node > 1 or cfg.num_nodes > 1:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
        mp.spawn(main_worker, nprocs=cfg.num_nodes * cfg.gpus_per_node, args=(cfg,))
    else:
        # Simply call main_worker function
        main_worker(cfg.device, cfg)

if __name__ == "__main__":
    main()
