export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node=4 train.py