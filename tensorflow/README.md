# PaddlePaddle Fluid 和 TensorFlow 并行特性对比

parallel computation | Fluid | Tensorflow
-- | -- | --
multiple devices | ParallelExecutor | parameter_server/replica
multiple nodes   | parameter_server/ring-base | distributed_replica/parameter_server/Horvod

## 多设备

### TensorFlow

TensorFlow 支持 `parameter_server` 和 `replica` 两种方式进行多卡的训练:

1. 在 `parameter_server` 模式中:
    - 训练进程中会保存一份主参数，通常是CPU Memory。
    - 每个设备称为一个 `worker`, 每个worker都保存有一个模型副本，在计算时会根据依赖关系从 `parameter_server` 中拉拉取需要的参数并做后续的计算。
    - 支持同步和异步训练.

1. 在 `replica` 模式中:
    - 每个设备拥有一个完整的模型副本以及参数的拷贝。
    - 每个设备独立计算 gradient后，使用 `Reduce/Broadcast` 来聚合/同步最新的参数同步到所有的设备上。
    - 支持同步训练。

### Fluid

Fluid 中使用 `ParallelExecutor` 支持多卡的并行训练:
- `SSAGraph Builder` 将用户配置的 ProgramDesc 转换成一个依赖图 `SSAGraph`.
- `ParallelExecutor` 会根据op的依赖关系并行的执行所有op。
- 和 TensorFlow 中 `replica` 模式采取一样的策略更新参数，并且在聚合参数时使用 NCCL2的 `Reduce/AllReduce` 来
聚合参数， 在大部分场景下 `Reduce` 性能会好一些。

## 多节点

### TensorFlow

1. parameter_server
    - 训练节点分为 `worker`, `ps` 两种角色，ps 中保存了参数的 master 副本。
    - worker 节点在训练时会根据依赖关系从 ps 中拉取最新的参数进行计算。
    - 支持同步和异步训练。
1. distribute_replica
    - 训练节点分为 `worker`,  `ps` 两种角色，但 ps 并不保存 master 副本。
    - 每个 worker 节点都保存一份完整的参数拷贝，并将计算出的 gradient 发送到 ps 节点进行更新，然后再将最新的参数同步回 worker 节点。
    - 支持同步和异步训练。
1. Horovrd
    - 无 `ps` 节点，所有参数均保存在 `worker` 节点中。
    - 每个 worker 节点占用一个 GPU 设备, 所有 worker 节点组成 Ring 结构。
    - 支持同步训练。

### Fluid

1. parameter_server
    - 训练节点分为 `trainer`, `ps` 两种角色。
    - trainer 节点保存完整的参数副本并计算 grad， ps 负责参数更新。
    - 支持同步，异步训练。
    - trainer 节点支持 prefetch 的方式从 pserver 拉取某一指定参数。
1. ring-base
    - 无 `ps` 节点，trainer 节点保存所有的参数拷贝。
    - 使用 NCCL2 的 Reduce/AllReduce/BroadCast 实现多GPU多节点之间的参数聚合和同步。
    - 只支持GPU的同步训练。
 
## Reference

- https://www.tensorflow.org/performance/performance_models
- https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
- https://github.com/uber/horovod