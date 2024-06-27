# BOAT

This is PyTorch implementation of [Boosting Consistency in Dual Training for Long-Tailed Semi-Supervised Learning](https://arxiv.org/pdf/2406.13187). Our conference paper published in CVPR 2023 is available at [ACR](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.pdf).

## Abstract

While long-tailed semi-supervised learning (LTSSL) has received tremendous attention in many real-world classification problems, existing LTSSL algorithms typically assume that the class distributions of labeled and unlabeled data are almost identical. Those LTSSL algorithms built upon the assumption can severely suffer when the class distributions of labeled and unlabeled data are mismatched since they utilize biased pseudo-labels from the model. To alleviate this problem, we propose a new simple method that can effectively utilize unlabeled data from unknown class distributions through Boosting cOnsistency in duAl Training (BOAT). Specifically, we construct the standard and balanced branch to ensure the performance of the head and tail classes, respectively. Throughout the training process, the two branches incrementally converge and interact with each other, eventually resulting in commendable performance across all classes. Despite its simplicity, we show that BOAT achieves state-of-the-art performance on a variety of standard LTSSL benchmarks, e.g., an averaged 2.7% absolute increase in test accuracy against existing algorithms when the class distributions of labeled and unlabeled data are mismatched. Even when the class distributions are identical, BOAT consistently outperforms many sophisticated LTSSL algorithms. We carry out extensive ablation studies to tease apart the factors that are the most important to the success of BOAT.

## Requirements

- Python 3.7.13
- PyTorch 1.12.0+cu116
- torchvision
- numpy

## Dataset

The directory structure for datasets looks like:

```
datasets
├── cifar-10
├── cifar-100
└── stl-10
```

## Usage

Train our proposed BOAT for different settings.

For CIFAR-10:

```
# run consistent setting
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --tau 1.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0

# run uniform setting
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 1 --tau 1.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0

# run reversed setting
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --tau 1.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0 --flag-reverse-LT 1
```

For CIFAR-100:

```
# run consistent setting
python train.py --dataset cifar100 --num-max 50 --num-max-u 400 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 10 --imb-ratio-unlabel 10 --tau 2.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0

# run uniform setting
python train.py --dataset cifar100 --num-max 50 --num-max-u 400 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 10 --imb-ratio-unlabel 1 --tau 2.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0

# run reversed setting
python train.py --dataset cifar100 --num-max 50 --num-max-u 400 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 10 --imb-ratio-unlabel 10 --tau 2.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0 --flag-reverse-LT 1
```

## Acknowledge

Our code of BOAT is based on the implementation of FixMatch. We thank the authors of the [FixMatch](https://github.com/kekmodel/FixMatch-pytorch) for making their code available to the public.

## Citation
For journal version: 
```
@article{gan2024boosting,
  title={Boosting Consistency in Dual Training for Long-Tailed Semi-Supervised Learning},
  author={Gan, Kai and Wei, Tong and Zhang, Min-Ling},
  journal={arXiv preprint arXiv:2406.13187},
  year={2024}
}
```
For conference version: 
```
@InProceedings{Wei_2023_CVPR,
    author    = {Wei, Tong and Gan, Kai},
    title     = {Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3469-3478}
}
```

