# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import argparse
import os
import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from matey import Trainer
from matey.utils import setup_dist, check_sp, profile_function, log_to_file, log_versions, YParams
import glob, socket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='00', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--use_fsdp", action='store_true', help='Use FullyShardedDataParallel')
    parser.add_argument("--yaml_config", default='./config/multi_ds.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--pei_debug", action='store_true', help='Pei debugging flag')
    parser.add_argument("--pei_oneloss", action='store_true', help='Pei debugging flag')
    parser.add_argument("--pei_filtered", action='store_true', help='Pei filtering flag')
    parser.add_argument("--pei_minres", action='store_true', help='Pei minimize residual flag')
    parser.add_argument("--pei_moduleloss", action='store_true', help='Pei optimize module loss flag')
    parser.add_argument("--pei_fixedupsample", action='store_true', help='Pei fix upsampling flag')
    parser.add_argument("--pei_linearupsample", action='store_true', help='Pei linear upsampling flag')
    parser.add_argument("--enable_sync", action='store_true', help='torch.cuda.synchronize flag')
    parser.add_argument("--enable_profiling", action='store_true', help='enable torch profiler flag')
    
    # Pipeline parallelism arguments
    parser.add_argument("--use_pipeline", action='store_true', 
                       help='Use pipeline parallelism with DeepSpeed')
    parser.add_argument("--pipeline_stages", type=int, default=2,
                       help="Number of pipeline parallel stages (default: 2)")
    parser.add_argument("--zero_stage", type=int, default=0,
                       help="ZeRO optimization stage (0, 1, 2, 3). Default is 0 (disabled).")

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = args.use_ddp
    params.use_fsdp = args.use_fsdp
    params.pei_debug = args.pei_debug
    params.pei_oneloss = args.pei_oneloss
    params.pei_filtered = args.pei_filtered
    params.pei_minres = args.pei_minres
    params.pei_moduleloss = args.pei_moduleloss
    params.enable_sync = args.enable_sync   
    params.profiling = args.enable_profiling
    
    # Add pipeline parameters
    params.use_pipeline = args.use_pipeline
    params.pipeline_stages = args.pipeline_stages
    params.zero_stage = args.zero_stage

    if not hasattr(params, "tokenizer_heads"):
        assert hasattr(params, "patch_size")
        params.tokenizer_heads=[{"head_name": "default",
                                 "patch_size": params.patch_size 
                                 }]
    print(params.tokenizer_heads, flush=True)
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] = args.pei_fixedupsample
        params.hierarchical["linearupsample"] = args.pei_linearupsample
        print(params.hierarchical, flush=True)
    
    # Set up distributed training
    device, world_size, local_rank, global_rank = setup_dist(params)
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}, host={socket.gethostname()}", flush=True)

    # Modify params
    params['batch_size'] = int(params.batch_size//world_size)
    params['startEpoch'] = 0
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))

    params['old_exp_dir'] = expDir
    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')

    # Have rank 0 check for and/or make directory
    if global_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)
    
    if params.use_fsdp:
        params['resuming'] = True if len(glob.glob(os.path.join(params.best_checkpoint_path, "*distcp"))) > 0 else False
    else:
        params['resuming'] = True if os.path.isfile(params.best_checkpoint_path) else False

    if params.pei_debug:
        params.debug_outdir = os.path.join(expDir, "./debug_outputs/")
        os.makedirs(params.debug_outdir, exist_ok=True)

    if global_rank == 0:
        log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        log_versions()
        params.log()

    params['log_to_screen'] = (global_rank == 0) and params['log_to_screen']
    torch.backends.cudnn.benchmark = False

    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)
    
    # ==========================================================
    # EXECUTION LOGIC: Choose which training mode to run
    # ==========================================================
    if params.use_pipeline:
        # Pipeline parallelism mode with DeepSpeed
        if global_rank == 0:
            print("=" * 80)
            print("STARTING PIPELINE PARALLELISM MODE")
            print(f"Pipeline stages: {params.pipeline_stages}")
            print(f"ZeRO stage: {params.zero_stage}")
            print(f"World size: {world_size}")
            print(f"Data parallel size: {world_size // params.pipeline_stages}")
            print("=" * 80)
        
        # Validation checks
        if world_size % params.pipeline_stages != 0:
            raise ValueError(f"World size ({world_size}) must be divisible by pipeline_stages ({params.pipeline_stages})")
        
        if params.use_ddp or params.use_fsdp:
            if global_rank == 0:
                print("WARNING: DDP/FSDP flags are ignored when using pipeline parallelism")
        
        # Disable sequence parallelism for pipeline mode
        if hasattr(params, "sp_groupsize") or hasattr(params, "num_sequence_parallel_groups"):
            if global_rank == 0:
                print("WARNING: Sequence parallelism is disabled in pipeline mode")
            # Force sequence parallel group size to 1 (no sequence parallelism)
            if hasattr(params, "sp_groupsize"):
                params.sp_groupsize = 1
            if hasattr(params, "num_sequence_parallel_groups"):
                params.num_sequence_parallel_groups = world_size // params.pipeline_stages
        
        # Import and initialize DeepSpeed distributed backend
        import deepspeed
        import torch.distributed as dist
        
        # DeepSpeed handles its own distributed initialization
        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend='nccl')
            if global_rank == 0:
                print("DeepSpeed distributed backend initialized")
        
        # Import pipeline training function
        try:
            from matey.pipeline_trainer_matey import train_with_pipeline_matey
        except ImportError:
            raise ImportError(
                "Pipeline training module not found. "
                "Make sure 'pipeline_trainer_matey.py' is in the matey package directory."
            )
        
        # Run pipeline training
        if global_rank == 0:
            print("Launching pipeline training...")
        
        train_with_pipeline_matey(params, global_rank, local_rank, world_size)
        
        if global_rank == 0:
            print("Pipeline training completed successfully!")
    
    else:
        # Original DDP/FSDP training path
        if global_rank == 0:
            print("=" * 80)
            print("STARTING STANDARD TRAINING MODE (DDP/FSDP)")
            if params.use_ddp:
                print("Using DistributedDataParallel (DDP)")
            elif params.use_fsdp:
                print("Using FullyShardedDataParallel (FSDP)")
            print(f"World size: {world_size}")
            print("=" * 80)
        
        # Standard trainer initialization
        trainer = Trainer(params, global_rank, local_rank, device)
        
        # Check if sequence parallel groups are defined properly
        check_sp(trainer.sequence_parallel_groups, global_rank)
        
        # Run training with optional profiling
        with profile_function(
            enabled=trainer.profiling, 
            logdir=os.path.join(expDir, "profiler_logs")
        ) as prof:
            trainer.train()
            if prof is not None:
                prof.step()
        
        if params.log_to_screen:
            print(f'DONE ---- rank {global_rank}')
