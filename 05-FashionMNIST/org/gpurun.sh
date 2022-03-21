rm gpulog
CUDA_VISIBLE_DEVICES=1 mpirun  -H localhost:1 -np 1 python mnist_fit.py 2>&1 |tee -a gpulog
