# Balanced data experiments

## Experiments on 2D Gaussians dataset

In order to reproduce our experiments, you can run the commands outlined below.
All unnamed hyperparameters have been set to the values used in the paper and can be examined in the respective files.

### Creating the dataset
To create the dataset, run the following command:
- `python3 synth_data_2d.py`

### Training and evaluation
To run the experiments, use the following settings

#### DP-MERF
epsilon = 1.0
- `python3 gen_balanced.py --data 2d -noise 5. --synth-spec-string norm_k5_n100000_row5_col5_noise0.2 -ep 50 --log-name dpmerf_syn2d_exp --gen-spec 200,500,500,200 --rff-sigma 0.50 -lr 3e-3 --d-rff 30000`

#### DP-CGAN
this code is found in the `../dpcgan/` directory.

non-private
- `python3 dp_cgan_synth2d.py --log-name dpcgan_syn2d_exp_nondp --epsilon 0. --noise 0. --clip 1000. -bs 500 --h-dim 512 --z-dim 100 --start-lr 0.03 --end-lr 0.01 --lr-saturate-epochs 10000 --num-training-steps 100000`

epsilon = 9.6
- `python3.6 dp_cgan_synth2d.py --log-name dpcgan_syn2d_exp_high_eps --epsilon 9.6 --noise 1. -bs 300 --clip 1. --h-dim 512 --z-dim 20 --start-lr 0.3 --end-lr 0.1 --lr-saturate-epochs 10000 --num-training-steps 100000`

epsilon = 1.0
- `python3.6 dp_cgan_synth2d.py --log-name dpcgan_syn2d_exp_low_eps --epsilon 1.0 --noise 5. -bs 300 --clip 1. --h-dim 1024 --z-dim 20 --start-lr 0.1 --end-lr 0.052 --lr-saturate-epochs 10000 --num-training-steps 100000`



## Experiments on MNIST

In order to reproduce our experiments, you can run the commands outlined below.
All hyperparameters have been set to the values used in the paper and can be examined in the respective files. Random seats for five runs used in the paper are 1-5
Please note, that DP-MERF downloads datasets, while the DP-CGAN code assumes they already exist, so make sure to run DP-MERF first.


## digit MNIST

#### DP-MERF
For the (1, 10^-5)-DP model, append `-noise 5.0` and for (0.2,10^-5)-DP, append `-noise 25.0`  
- `python3 mnist_sr_gen.py --log-name dpmerf_digits_exp --data digits`

#### DP-CGAN
this code is found in the `../dpcgan/` directory.

- `python3 dp_cgan_reference_im.py --data-save-str dpcgan_digits --data digits`

#### DP-GAN
this code is found in the `../dpgan/` directory.
- `python3 class_wise_wgan.py --n-epochs 20 --dp-noise 1.41 --log-name dpgan_digits_exp --data digits`

#### GS-WGAN
this code is found in the `../gs-wgan/` directory. Prior to training, we download the pre-trained discriminators as described in `../gs-wgan/README.md` 
- `python3.6 main.py -data 'mnist' -name 'gswgan_digits_exp' -ldir '../models/ResNet_default' -noise 1.07`

### Evaluation
each of the above models creates a synthetic dataset, which can be evaluated by running the following script with the previously used experiment name (`log-name` or `data-save-str`)
- `python3 synth_data_benchmark.py --data-log-name *experiment name* --data_base_dir *relative path to data dir* --data digits`

Due to slight differences in data loading, the GS-WGAN model is evaluated through `../gs-wgan/gs-wgan-eval.py` which internally calls `synth_data_benchmark`.

## fashion MNIST

All experiments are run with the same hyperparameters. The only change required is switching the `--data` flag to `fashion`
