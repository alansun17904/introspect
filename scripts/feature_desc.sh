#!/bin/bash
#
#SBATCH --mem=128G
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres gpu:2            # set 2 GPUs per job
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 0-12:00              # Maximum run-time in D-HH:MM
#SBATCH -o feature_desc_stdout_%j.out      # File to which STDOUT will be written
#SBATCH -e feature_desc_stderr_%j.err      # File to which STDERR will be written

source /BRAIN/circuit-alignment/work/cache/virtualenvs/venv/bin/activate
source .env

python -m torch.distributed.run --nproc_per_node=2 -m introspect feature train --config configs/gemma-scope-features.yaml