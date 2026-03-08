#!/bin/bash
#SBATCH -A stf218
#SBATCH -J matey-mode2
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -C nvme
##SBATCH -q debug

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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Mode 2 test: single GPU, no distributed, no SLURM multi-process.
# Verifies that pipeline_forward_sequential (stage-by-stage with packed
# tensors) produces identical output to TurbTIterative.forward().
srun -N1 -n1 -c7 --gpus=1 python test_pipeline_equivalence.py \
    --yaml_config ./config/Demo_JHUTDB_TT.yaml --config basic_config
