#!/bin/bash

JOB_NAME=worker
TASK_INDEX=$1
PS_HOSTS=192.168.16.25:9001,192.168.16.26:9001,192.168.16.29:9001,192.168.16.30:9001
WORKER_HOSTS=192.168.16.25:8001,192.168.16.26:8001,192.168.16.29:8001,192.168.16.30:8001

stdbuf -oL nohup python tf_cnn_benchmarks.py \
--data_format=NCHW --batch_size=64 \
--model=resnet50 --optimizer=momentum \
--variable_update=distributed_replicated \
--nodistortions --gradient_repacking=8 \
--num_gpus=8 \
--num_epochs=360 \
--data_name=imagenet \
--weight_decay=1e-4  \
--print_training_accuracy \
--nodistortions \
--data_dir=./output \
--job_name=$JOB_NAME \
--ps_hosts=$PS_HOSTS \
--worker_hosts=$WORKER_HOSTS \
--task_index=$TASK_INDEX \
--datasets_num_private_threads=16 \
--device gpu \
--allow_growth  2>&1 > ${JOB_NAME}${TASK_INDEX}.log &