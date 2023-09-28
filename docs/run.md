# Training Instructions

## Training with different benchmarks
Note: We train each of our method for the all the 3 benchmarks.
To train with different benchmarks, change the config parameter argument with the following choices.
- For Benchmark 1 choose values for config from [M, C, I, O]
- For Benchmark 2 choose values for config [cefa, surf, wmca]
- For Benchmark 3 choose values for config  [CI, CO , CM, MC, MI, MO, IC, IO, IM, OC, OI, OM]


## Training FLIP-V
```shell
python train_flip.py \
        --op_dir path/to/output/directory/ \
        --report_logger_path path/to/save/performance.csv \
        --config wmca \
        --method flip_v
```

## Training FLIP-IT
```shell
python train_flip.py \
        --op_dir path/to/output/directory/ \
        --report_logger_path path/to/save/performance.csv \
        --config cefa \
        --method flip_it
```


## Training FLIP-MCL
```shell
python train_flip_mcl.py \
        --op_dir path/to/output/directory/ \
        --report_logger_path path/to/save/performance.csv \
        --config M
```


## Inference
Choose between the following methods for inference: flip_v, flip_it, flip_mcl
```shell
python infer_flip.py \
        --op_dir path/to/output/directory/ \
        --report_logger_path path/to/save/performance.csv \
        --ckpt path/to/checkpoint.pth \
        --config M \
        --method flip_mcl
```

TODO: Integrate all the training scripts into one file and add configurable arguments to choose the method and benchmark more easily.


### 0-shot, 5-shot and Benchmarks
To train in 0-shot and 5-shot setting uncomment the corresponding line in each file, as per the instructions.
Similalry, to train on the three separate benchmarks, uncomment the corresponding line in each file, as per the instructions.