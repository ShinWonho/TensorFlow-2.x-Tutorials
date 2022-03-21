rm 4gpulog
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun  -H localhost:4 -np 4 python mnist_fit.py 2>&1 |tee -a 4gpulog
