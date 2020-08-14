#!/bin/bash
#SBATCH -J CUDA
#SBATCH -A Project
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o CUDA.out
#SBATCH -e CUDA.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<e-mail address>
for b in 32 64 128 256
do
	/usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$b -o CUDA CUDA.cu
	./CUDA
done
