"""Trainer for OCR CTC model."""
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_train_net
import ctc_reader
import argparse
import functools
import sys
import time
import os
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.transpiler.distribute_transpiler as distribute_transpiler
ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('pass_num',          int,   100,        "Number of training epochs.")
add_arg('log_period',        int,   1000,       "Log period.")
add_arg('save_model_period', int,   15000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   15000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./models", "The directory the model to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('min_average_window',int,   10000,     "Min average window.")
add_arg('max_average_window',int,   15625,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,      "Average window.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('local',             bool,  True,      "Local train or distributed.")
# yapf: enable

def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('Total examples: %d, total time: %.5f, performance: %.5f examples/sed' %
          (num_samples, train_elapsed, examples_per_sec))

def train(args, data_reader=ctc_reader):
    """OCR CTC training"""
    num_classes = None
    train_images = None
    train_list = None
    test_images = None
    test_list = None
    num_classes = data_reader.num_classes(
    ) if num_classes is None else num_classes
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)
    sum_cost, error_evaluator, inference_program, model_average = ctc_train_net(
        images, label, args, num_classes)

    # data reader
    train_reader = data_reader.train(
        args.batch_size,
        train_images_dir=train_images,
        train_list_file=train_list)
    test_reader = data_reader.test(
        test_images_dir=test_images, test_list_file=test_list)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    if args.init_model is not None:
        model_dir = args.init_model
        model_file_name = None
        if not os.path.isdir(args.init_model):
            model_dir = os.path.dirname(args.init_model)
            model_file_name = os.path.basename(args.init_model)
        fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
        print "Init model from: %s." % args.init_model

    train_exe = exe
    error_evaluator.reset(exe)
    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=True, loss_name=sum_cost.name)

    fetch_vars = [sum_cost] + error_evaluator.metrics

    def test(pass_id, batch_id):
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        print("Pass[%d]-batch[%d]; Test seq error: %s." % (
              pass_id, batch_id, str(test_seq_error[0])))

    def save_model(args, exe, pass_id, batch_id):
        filename = "model_%05d_%d" % (pass_id, batch_id)
        fluid.io.save_params(
            exe, dirname=args.save_model_dir, filename=filename)
        print "Saved model to: %s/%s." % (args.save_model_dir, filename)

    def train_parallel(train_exe):
        var_names = [var.name for var in fetch_vars]
        for pass_id in range(args.pass_num):
            batch_id = 1
            total_loss = 0.0
            total_seq_error = 0.0
            # train a pass
            num_samples, start_time = 0, time.time()
            for data in train_reader():
                batch_start_time = time.time()
                results = train_exe.run(var_names, feed=get_feeder_data(data, place))
                results = [np.array(result).sum() for result in results]
                total_loss += results[0]
                total_seq_error += results[2]
                # training log
                if batch_id % args.log_period == 0:
                    print("Pass[%d]-batch[%d]; Avg Warp-CTC loss: %s; Avg seq err: %s; Speed: %.5f samples/sec" % (
                          pass_id, batch_id,
                          total_loss / (batch_id * args.batch_size),
                          total_seq_error / (batch_id * args.batch_size),
                          len(data) / (time.time() - batch_start_time)))

                # evaluate
                
                if batch_id % args.eval_period == 0:
                    if model_average:
                        with model_average.apply(exe):
                            test(pass_id, batch_id)
                    else:
                        test(pass_id, batch_id)

                # save model
                """
                if batch_id % args.save_model_period == 0:
                    if model_average:
                        with model_average.apply(exe)
                            save_model(args, exe, pass_id, batch_id)
                    else:
                        save_model(args, exe, pass_id, batch_id)
                """
                batch_id += 1
                train_exe.bcast_params()
                num_samples += len(data)
            print_train_time(start_time, time.time(), num_samples)
  
    if args.local:
        place = core.CPUPlace() if args.use_gpu else core.CUDAPlace(0)
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        exec_strategy = ExecutionStrategy()
        exec_strategy.use_cuda = args.use_gpu
        train_exe = fluid.ParallelExecutor(use_cuda=args.use_gpu, main_program=fluid.default_main_program(), loss_name=sum_cost.name, exec_strategy=exec_strategy)
        train_parallel(train_exe)
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        # the IP of the local machine, needed by pserver only
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
        # the role, should be either PSERVER or TRAINER
        training_role = os.getenv("PADDLE_TRAINING_ROLE")
        t = distribute_transpiler.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        place = core.CPUPlace() if args.use_gpu else core.CUDAPlace(0)
        if training_role == "PSERVER":
            pserver_program = t.get_pserver_program(current_endpoint)
            pserver_startup_program = t.get_startup_program(current_endpoint,
                                                        pserver_program)
            exe = fluid.Executor(place)
            exe.run(pserver_startup_program)
            exe.run(pserver_program)
        elif training_role == "TRAINER":
            startup_exe = fluid.Executor(place)
            startup_exe.run(fluid.default_startup_program())
            trainer_program = t.get_trainer_program()
            exec_strategy = ExecutionStrategy()
            exec_strategy.use_cuda = args.use_gpu
            train_exe = fluid.ParallelExecutor(use_cuda=args.use_gpu, main_program=trainer_program, loss_name=sum_cost.name, exec_strategy=exec_strategy)
            train_parallel(train_exe)
        else:
            raise ValueError("env PADDLE_TRAINING_ROLE should be in [PSERVER, TRIANER]")
        



def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args, data_reader=ctc_reader)


if __name__ == "__main__":
    main()
