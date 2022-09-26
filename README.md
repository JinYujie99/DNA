# DNA: Domain Generalization with Diversified Neural Averaging

PyTorch implementation of "DNA: Domain Generalization with Diversified Neural Averaging". Our work is built upon [SWAD](https://github.com/khanrc/swad), which is released under the MIT license.

## Usage
1. Dependencies
```sh
pip install -r requirements.txt
```

2. Download the datasets
```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```
3. Run training

`train_all.py` script launches multiple leave-one-out experiments, each of which treats one domain as the target domain.

For example, you can run the following instructions to launch 3 runs with different random dataset splits on TerraIncognita (with the default hyperparameters).
```
python train_all.py TR0 --dataset TerraIncognita --deterministic --trial_seed 0  --data_dir /my/datasets/path
python train_all.py TR1 --dataset TerraIncognita --deterministic --trial_seed 1  --data_dir /my/datasets/path
python train_all.py TR2 --dataset TerraIncognita --deterministic --trial_seed 2  --data_dir /my/datasets/path
```
The results are reported as a table. In the table, the row `SWAD` indicates out-of-domain accuracy of the ensemble model, and the row `SWAD(inD)` indicates the in-domain validation accuracy.

To reproduce the results of DNA, we list the recommended hyperparameters searched by us in hparams_registry.py. You can also manually search hyperparameters by modifying them in CLI. For example, you can set dropout_rate to 0.1 by adding `--dropout_rate 0.1`.

## Requirements

Environment details used for our experiments.

```
Python: 3.7.9
PyTorch: 1.7.1
Torchvision: 0.8.2
CUDA: 11.2
CUDNN: 7605
NumPy: 1.19.4
PIL: 8.0.1
```






