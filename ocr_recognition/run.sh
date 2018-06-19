#!/bin/bash

function start_pserver() {
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=PSERVER python ctc_train.py --local 0
}

function start_train() {
  CUDA_VISIBLE_DEVICES=2,3 \
  FLAGS_fraction_of_gpu_memory_to_use=0.8 \
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=TRAINER python ctc_train.py --local 0
}