
# Single Card

## Help

```angular2html
$ python main.py --help
usage: main.py [-h] [--batch-size N] [--test-batch-size N] [-e N] [--log-interval N] [--dry-run]

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 64)
  -e N, --epochs N     number of total epochs to run (default: 2)
  --log-interval N     how many batches to wait before logging training status (default: 10)
  --dry-run            quickly check a single pass (default: true)
```

## Run

```angular2html
$ python main.py -e 2
[PID 2901799]	Train Epoch: 0 [0/60000 (0%)]	Loss: 2.698152
[PID 2901799]	Train Epoch: 0 [640/60000 (1%)]	Loss: 2.618320
[PID 2901799]	Train Epoch: 0 [1280/60000 (2%)]	Loss: 2.563968
[PID 2901799]	Train Epoch: 0 [1920/60000 (3%)]	Loss: 2.420666
...
...
...
[PID 2901799]	Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.803352
[PID 2901799]	Train Epoch: 1 [58240/60000 (97%)]	Loss: 0.823862
[PID 2901799]	Train Epoch: 1 [58880/60000 (98%)]	Loss: 0.765787
[PID 2901799]	Train Epoch: 1 [59520/60000 (99%)]	Loss: 0.854648
Training one epoch in: 0:00:11.394154
Training complete in: 0:00:22.391556

Test set: Average loss: 0.0135, Accuracy: 8783/10000 (88%)

Training one epoch in: 0:00:02.402262
```