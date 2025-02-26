# Schur's Positive-Definite Network

This repository contains our PyTorch implementation of the SpodNet model described in `Schur's Positive-Definite Network: Deep Learning in the SPD cone with structure. Can Pouliquen, Mathurin Massias, Titouan Vayer (ICLR 2025)`

## Preamble

In order to launch the scripts, first run the `pip install -e .` command from the current location (`./`).

## Data generation

Run the `./exps/data/generate_dataset.py` script to generate training and testing sets.
The following two commands are generic examples:

```
python generate_dataset.py -set_type train -n 100 -p 20 -den 0.95 -size 10000 -as_tensor 1 -random_state 0 -same_Theta_true 0 ;
```
```
python generate_dataset.py -set_type test -n 100 -p 20 -den 0.95 -size 10000 -as_tensor 1 -random_state 1 -same_Theta_true 0 ;
```

## SpodNet implementation

`SpodNet` layers are contained in the `./spodnet/` folder.

* `./spodnet/framework.py` contains the main `SpodNet` class, which is the generic layer performing various possible updates to create the `UBG`, `PNP` or `E2E` architectures (specified during initialization of the class).

* `./spodnet/perturbation_layers.py` contains the classes implementing the three models' different update rules.

## Training

The script for training the different models on various datasets is in `./exps/train_spodnet.py`, with the desired setting that can be specified in arguments when launching the script.
After running the two above lines to generate train and test datasets, a typical command to do so is for example:

```
python train_spodnet.py -train_samples 1000 -test_samples 100 -n 100 -p 20 -train_batch_size 10 -test_batch_size 100 -precision_sparsity 0.95 -K 1 -epochs 20 -lr 1e-2 -learning_mode 'UBG'
```
