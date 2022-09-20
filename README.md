# On the Tradeoff between Robustness and Fairness

Code for NeurIPS'22 paper, **"On the Tradeoff between Robustness and Fairness"**  by Xinsong Ma, Zekai Wang, Weiwei Liu.

### Preferred Prerequisites

* Python (3.8)
* Pytorch (1.9.0)
* numpy (1.21)
* tensorboard (2.7)
* CUDA (11.1)

### Simple Instructions

#### Train Resnet50 on CIFAR 100 using AT(Madry) with perturbation raius=0.015

```bash
CUDA_VISIBLE_DEVICES='0' python pgd_var.py --epsilon 0.015 --net resnet50 --dataset cifar100 --seed 1 --batch_size 256 --no_regular
```

#### Use our FAT on AT(Madry) with regularization = 0.1

```bash
CUDA_VISIBLE_DEVICES='0' python pgd_var.py --epsilon 0.015 --net resnet50 --dataset cifar100 --seed 1 --batch_size 256 --alpha 0.1
```

#### Train Resnet18 on CIFAR 10 using original TRADES with beta=6.0, perturbation raius=0.062

```bash
CUDA_VISIBLE_DEVICES='0' python trades_var.py --epsilon 0.062 --net resnet18 --beta 6.0 --dataset cifar10 --batch_size 256 --seed 1 --no_regular
```

#### Use our FAT on TRADES with regularization = 0.1

```bash
CUDA_VISIBLE_DEVICES='0' python trades_var.py --epsilon 0.062 --net resnet18 --beta 6.0 --dataset cifar10 --batch_size 256 --seed 1  --alpha 0.1
```


#### Evaluate models using FGSM, PGD and CW attack

```bash
CUDA_VISIBLE_DEVICES='0' python var_acc_test.py --attack_type fgsm --dataset 'cifar10' --dir_path './test_models'
```
```bash
CUDA_VISIBLE_DEVICES='0' python var_acc_test.py --attack_type pgd20 --dataset 'cifar10' --dir_path './test_models'
```
```bash
CUDA_VISIBLE_DEVICES='0' python var_acc_test.py --attack_type cw --dataset 'cifar10' --dir_path './test_models'
```

You can replace the device parameter 'CUDA_VISIBLE_DEVICES' and parameter '-- epsilon', '-- seed' if you need.