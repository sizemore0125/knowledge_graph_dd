#!/bin/bash
#SBATCH --job-name=kgdd
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --requeue
#SBATCH --partition=ampere

set -e
module purge
module load slurm
module load python/3.10
module load cuda

cd /nfs/stak/users/sizemolo/kgdd/knowledge_graph_dd
source .venv/bin/activate

which python
python --version

nvidia-smi

echo "Starting Script"
python main.py
echo "Script Ended"
