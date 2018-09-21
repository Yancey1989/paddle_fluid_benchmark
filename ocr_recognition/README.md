# Distributed OCR Model Training

This folder contains implement of OCR model which supported distributed training, you can check out
the PaddlePaddle offical [OCR Recognition Model](https://github.com/PaddlePaddle/models/tree/develop/fluid/ocr_recognition) to know more abouht the introduce
of the OCR model.

## Get Started

The entrypoint file is `dist_train.py`, you can get the usage by `python dist_train.py -h`, an example
command to launch a distributed training job with 2 parameter server instances and 2 trainer instances is as follows:

1. launch the pserver process

    ``` python
      PADDLE_PSERVER_PORT=6173 \
      PADDLE_PSERVER_IPS=192.168.0.100,192.168.0.101 \
      PADDLE_TRAINERS=1 \
      PADDLE_CURRENT_IP=192.168.0.100 \
      PADDLE_TRAINING_ROLE=PSERVER \
      stdbuf -oL nohup python ctc_train.py --local 0 --use_gpu 0
    ```

1. launch trainer process

    ``` python
      PADDLE_PSERVER_PORT=6173 \
      PADDLE_TRAINER_ID=0 \
      PADDLE_PSERVER_IPS=192.168.0.100,192.168.0.101 \
      PADDLE_TRAINERS=2 \
      PADDLE_TRAINING_ROLE=TRAINER \
      stdbuf -oL nohup python ctc_train.py --batch_size 256 --eval_period 500 --log_period 100 --local 0
    ```
