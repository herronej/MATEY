#!/bin/bash
#SBATCH -A stf218
#SBATCH -J matey-mode3
#SBATCH -o %x-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH -C nvme

export OMP_NUM_THREADS=1

module load miniforge3/23.11.0
module load gcc/12.2.0
module load rocm/6.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/stf218/world-shared/emily/MATEY/matey_env_3_10
module unload miniforge3/23.11.0
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
##export NCCL_DEBUG=INFO

# Mode 3: distributed pipeline parallel test.
# nlevels=3 -> 3 pipeline groups × 3 stages = 9 ranks.
# Inter-stage communication uses chunked broadcast (NCCL-safe).
srun -N$SLURM_JOB_NUM_NODES -n9 -c7 --gpu-bind=closest \
    python test_pipeline_mode3.py \
    --yaml_config ./config/Demo_JHUTDB_TT.yaml --config basic_config
