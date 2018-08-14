#!/bin/bash
set -x

stdbuf -oL nohup mpirun -np 32 \
-H instance-5af93isk:8,instance-8hqmi70e:8,instance-fjhn02ur:8,instance-927obypm:8 \
-x NCCL_DEBUG=DEBUG -x LD_LIBRARY_PATH -x PATH \
-x NCCL_SOCKET_IFNAME=^virbr0 \
-x NCCL_IB_DISABLE=1 \
-x HOROVOD_MPI_THREADS_DISABLE=1 \
-bind-to none -map-by slot \
-mca btl_tcp_if_exclude lo,virbr0 \
-mca plm_rsh_args "-p 12345" \
-mca pml ob1 \
/bin/bash -c '
stdbuf -oL python /work/tf_cnn_benchmarks.py \
--data_name=imagenet \
--data_format=NHWC --batch_size=64 \
--model=resnet50 --optimizer=momentum \
--datasets_num_private_threads=16 \
--weight_decay=1e-4  \
--datasets_prefetch_buffer_size=128 \
--data_dir=/work/output \
--print_training_accuracy \
--device gpu \
--variable_update=horovod \
--num_epochs 120 2>&1 > /work/horover.log.$OMPI_COMM_WORLD_LOCAL_RANK' 2>&1 > lunch_horvod.log &