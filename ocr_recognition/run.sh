#!/bin/bash
export PYTHONPATH=/paddle/build/python
export LD_LIBRARY_PATH=/paddle/build/python/paddle/libs/:$LD_LIBRARY_PATH
unset http_proxy
unset https_proxy

function start_pserver() {
  CUDA_VISIBLE_DEVICES=0 \
  GLOG_v=3 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=PSERVER \
  stdbuf -oL nohup python dist_train.py --local 0 --use_gpu 0 2>&1 > pserver_0.log &
}

function start_trainer() {
  FLAGS_fraction_of_gpu_memory_to_use=0.8 \
  GLOG_v=0 \
  GLOG_logtostderr=1 \
  PADDLE_PSERVER_PORT=6173 \
  PADDLE_TRAINER_ID=0 \
  PADDLE_PSERVER_IPS=127.0.0.1 \
  PADDLE_TRAINERS=1 \
  PADDLE_CURRENT_IP=127.0.0.1 \
  PADDLE_TRAINING_ROLE=TRAINER \
  stdbuf -oL nohup python dist_train.py --batch_size 256 --eval_period 500 --log_period 100 --local 0 2>&1 > trainer_0.log &
}

function start_local() {
  FLAGS_fraction_of_gpu_memory_to_use=0.8 \
  GLOG_v=0 \
  GLOG_logtostderr=1 \
  stdbuf -oL nohup python dist_train.py --batch_size 256 --log_period 100 --eval_period 500 --local 1 2>&1 > local_train.log &
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
    ;;
  *)
    echo "arg[0] should be in [pserver, trainer, local]"
    ;;
esac