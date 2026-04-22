#!/bin/bash
#SBATCH --job-name=distill_interactive
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/interactive_%j.out
#SBATCH --error=logs/interactive_%j.err

sleep infinity