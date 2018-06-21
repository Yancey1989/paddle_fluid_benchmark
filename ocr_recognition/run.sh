#!/bin/bash

function start_pserver() {
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=PSERVER \
  stdbuf -oL nohup python ctc_train.py --local 0 --use_gpu 0 2>&1 > pserver_0.log &
}

function start_trainer() {
  CUDA_VISIBLE_DEVICES=2,3 \
  FLAGS_fraction_of_gpu_memory_to_use=0.8 \
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=TRAINER \
  stdbuf -oL python python ctc_train.py --local 0 2>&1 > trainer_0.log &
}

function start_local() {
  CUDA_VISIBLE_DEVICES=0,1 \
  FLAGS_fraction_of_gpu_memory_to_use=0.8 \
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  stdbuf -oL nohup python ctc_train.py --local 1 2>&1 > local_train.log &
}

function stop() {
  kill `ps -ef |grep python ctc_train.py | awk '{print $2}'`
}

case $1 in
  "pserver")
    start_pserver
    ;;
  "trainer")
    start_trainer
    ;;
  "local")
    start_local
    ;;
  "stop")
    stop
  *)
    echo "arg[0] should be in [pserver, trainer, local]"
    ;;
esac