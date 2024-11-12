 export CUDA_VISIBLE_DEVICES=2

 torchrun --nproc_per_node 1 noise_test.py