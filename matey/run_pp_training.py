# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
import gc
import psutil
from torchinfo import summary
from collections import defaultdict
import json
from mpi4py import MPI
import sys
import glob
import random
import numpy as np
from train import Trainer
# DeepSpeed for pipeline parallelism
import deepspeed

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict, set_optimizer_state_dict

if __name__ == '__main__':
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='00', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--use_fsdp", action='store_true', help='Use FullyShardedDataParallel')
    parser.add_argument("--yaml_config", default='./config/multi_ds.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    
    # Pipeline parallelism arguments
    parser.add_argument("--use_pipeline", action='store_true', 
                       help='Use pipeline parallelism with DeepSpeed')
    parser.add_argument("--pipeline_stages", type=int, default=2,
                       help="Number of pipeline parallel stages")
    parser.add_argument("--zero_stage", type=int, default=0,
                       help="ZeRO optimization stage (0, 1, 2, 3). Default is 0 (disabled).")

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = args.use_ddp
    params.use_fsdp = args.use_fsdp
    params.use_pipeline = args.use_pipeline
    params.pipeline_stages = args.pipeline_stages
    params.zero_stage = args.zero_stage

    # Handle tokenizer heads
    if not hasattr(params, "tokenizer_heads"):
        assert hasattr(params, "patch_size")
        params.tokenizer_heads = [{
            "head_name": "default",
            "patch_size": params.patch_size
        }]
    print(params.tokenizer_heads, flush=True)

    # Set up distributed training with MPI
    num_gpus_per_node = torch.cuda.device_count()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(num_gpus_per_node) if num_gpus_per_node > 0 else 0
    
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'
    
    if os.getenv("SLURM_STEP_NODELIST") is not None:
        os.environ['MASTER_ADDR'] = parse_slurm_nodelist(os.environ["SLURM_STEP_NODELIST"])[0]

    # ==========================================================
    # EXECUTION LOGIC: Choose which training mode to run
    # ==========================================================
    if params.use_pipeline:
        # DeepSpeed pipeline parallelism path
        if global_rank == 0:
            print("=" * 60)
            print("USING DEEPSPEED PIPELINE PARALLELISM")
            print(f"Pipeline stages: {params.pipeline_stages}")
            print(f"ZeRO stage: {params.zero_stage}")
            print(f"World size: {world_size}")
            print("=" * 60)
        
        # DeepSpeed initialization is handled inside train_with_pipeline_matey
        deepspeed.init_distributed(dist_backend='nccl')
        
        # Modify params for pipeline mode
        params['batch_size'] = int(params.batch_size // world_size)
        params['startEpoch'] = 0
        
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))
        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
        
        if global_rank == 0:
            if not os.path.isdir(expDir):
                os.makedirs(expDir)
                os.makedirs(os.path.join(expDir, 'training_checkpoints/'))
        
        params['resuming'] = False  # Pipeline doesn't support checkpoint resume yet
        params['name'] = str(args.run_name)
        params['log_to_wandb'] = False  # Disable wandb for pipeline
        params['log_to_screen'] = (global_rank == 0)
        
        if global_rank == 0:
            logging_utils.log_to_file(logger_name=None, 
                                     log_filename=os.path.join(expDir, 'out.log'))
            logging_utils.log_versions()
            params.log()
            
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams, hpfile)
        
        torch.backends.cudnn.benchmark = False
        
        # Run pipeline training
        train_with_pipeline_matey(params, global_rank, local_rank, world_size)
    
    else:
        # Original DDP/FSDP path
        if global_rank == 0:
            print("=" * 60)
            print("USING STANDARD DDP/FSDP TRAINING")
            print(f"DDP: {params.use_ddp}, FSDP: {params.use_fsdp}")
            print(f"World size: {world_size}")
            print("=" * 60)
        
        if params.use_ddp or params.use_fsdp:
            dist.init_process_group(
                backend="nccl",
                init_method='env://',
                rank=rank,
                world_size=world_size,
            )
            torch.cuda.set_device(local_rank)
        
        device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
        print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")

        # Modify params
        params['batch_size'] = int(params.batch_size // world_size)
        params['startEpoch'] = 0
        
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))
        params['old_exp_dir'] = expDir
        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
        params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')

        if global_rank == 0:
            if not os.path.isdir(expDir):
                os.makedirs(expDir)
                os.makedirs(os.path.join(expDir, 'training_checkpoints/'))
        
        if params.use_fsdp:
            params['resuming'] = True if len(glob.glob(os.path.join(params.checkpoint_path, "*distcp"))) > 0 else False
        else:
            params['resuming'] = True if os.path.isfile(params.checkpoint_path) else False

        params['name'] = str(args.run_name)
        
        if global_rank == 0:
            logging_utils.log_to_file(logger_name=None, 
                                     log_filename=os.path.join(expDir, 'out.log'))
            logging_utils.log_versions()
            params.log()

        params['log_to_wandb'] = False  # Set to True if you want W&B
        params['log_to_screen'] = (global_rank == 0) and params['log_to_screen']
        torch.backends.cudnn.benchmark = False

        if global_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams, hpfile)
        
        # Create trainer and train
        trainer = Trainer(params, global_rank, local_rank, device)
        trainer.train()

    if params.log_to_screen:
        print('DONE ---- rank %d' % global_rank)
