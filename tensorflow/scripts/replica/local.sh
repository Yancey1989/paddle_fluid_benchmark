#!/bin/bash


stdbuf -oL nohup python tf_cnn_benchmarks.py \
--data_format=NHWC --batch_size=64 \
--model=resnet50 --optimizer=momentum \
--variable_update=replicated \
--num_gpus=8 --num_epochs=120 \
--data_name=imagenet \
--datasets_prefetch_buffer_size=128 \
--weight_decay=1e-4  \
--datasets_num_private_threads=16 \
--data_dir=./output \
--print_training_accuracy \
--device gpu 2>&1 > local_train.log &