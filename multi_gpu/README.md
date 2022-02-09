
# Multi Card

There are two ways to launch multi-distributed program: spawn mode or launch mode. Although pytorch provides many [examples](https://github.com/pytorch/examples) for using spawn mode, many projects use launch mode. And it looks launch mode is better than spawn mode, so I also choose to switch it.

## Help

```angular2html
$ python -m torch.distributed.launch --nproc_per_node=2 main.py --help
/home/zj/miniconda3/envs/py38/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
usage: main.py [-h] [--local_rank LOCAL_RANK] [--batch-size N] [--test-batch-size N] [-e N]
               [--log-interval N] [--dry-run]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        Local rank. Necessary for using the torch.distributed.launch utility. (default:
                        0)
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 64)
  -e N, --epochs N      number of total epochs to run (default: 2)
  --log-interval N      how many batches to wait before logging training status (default: 10)
  --dry-run             quickly check a single pass (default: true)
usage: main.py [-h] [--local_rank LOCAL_RANK] [--batch-size N] [--test-batch-size N] [-e N]
               [--log-interval N] [--dry-run]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        Local rank. Necessary for using the torch.distributed.launch utility. (default:
                        0)
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 64)
  -e N, --epochs N      number of total epochs to run (default: 2)
  --log-interval N      how many batches to wait before logging training status (default: 10)
  --dry-run             quickly check a single pass (default: true)
```

## Run

```angular2html
$ python -m torch.distributed.launch --nproc_per_node=2 main.py -e 2
/home/zj/miniconda3/envs/py38/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
0 cuda:0 2
1 cuda:1 2
[PID 3560620]	Train Epoch: 0 [0/60000 (0%)]	Loss: 2.315692
[PID 3560620]	Train Epoch: 0 [640/60000 (2%)]	Loss: 2.286061
[PID 3560620]	Train Epoch: 0 [1280/60000 (4%)]	Loss: 2.292775
[PID 3560620]	Train Epoch: 0 [1920/60000 (6%)]	Loss: 2.352640
[PID 3560620]	Train Epoch: 0 [2560/60000 (9%)]	Loss: 2.240308
[PID 3560620]	Train Epoch: 0 [3200/60000 (11%)]	Loss: 2.244782
...
...
[PID 3560620]	Train Epoch: 1 [28800/60000 (96%)]	Loss: 1.350578
[PID 3560620]	Train Epoch: 1 [29440/60000 (98%)]	Loss: 1.343520
Training one epoch in: 0:00:05.560163
Training complete in: 0:00:11.476325
Accuracy on all data: 0.7876999974250793, accelerator rank: 0
Accuracy on all data: 0.7876999974250793, accelerator rank: 1
Training one epoch in: 0:00:00.801581
```