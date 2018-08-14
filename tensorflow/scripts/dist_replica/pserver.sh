#!/bin/bash

JOB_NAME=ps
TASK_INDEX=$1
PS_HOSTS=192.168.16.25:9001,192.168.16.26:9001,192.168.16.29:9001,192.168.16.30:9001
WORKER_HOSTS=192.168.16.25:8001,192.168.16.26:8001,192.168.16.29:8001,192.168.16.30:8001

stdbuf -oL nohup python tf_cnn_benchmarks.py \
--local_parameter_device=gpu \
--num_gpus=8 \
--optimizer=momentum \
--batch_size=64 \
--model=resnet50 \
--variable_update=distributed_replicated \
--num_epochs=360 \
--job_name=$JOB_NAME \
--ps_hosts=$PS_HOSTS \
--worker_hosts=$WORKER_HOSTS \
--task_index=$TASK_INDEX \
--allow_growth  2>&1 > ${JOB_NAME}${TASK_INDEX}.log &