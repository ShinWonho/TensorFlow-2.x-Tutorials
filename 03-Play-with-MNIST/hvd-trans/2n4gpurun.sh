rm 2n4gpulog
# TRACE, DEBUG
#export HOROVOD_LOG_LEVEL=TRACE
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/openmpi/bin/mpirun  -H dl019:4,dl020:4 -np 8 -mca btl_opind-to none -map-by socket -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CUDA_VISIBLE_DEVICES -x HOROVOD_LOG_LEVEL -mca pml ob1 -mca btl_openib_allow_ib 1 python main.py 2>&1 |tee -a 2n4gpulog
