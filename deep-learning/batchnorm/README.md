# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

ICML 2015 | [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c81ca93fdd9c4a54b53df) | [Paper](https://arxiv.org/abs/1502.03167)

## What is in the code
- Batch normalization written from scratch: mini-batch statistics in training, running averages at inference, learnable gamma and beta
- Numerical check against torch.nn.BatchNorm1d
- 6-layer sigmoid MLP on scikit-learn digits with and without BN

## Differences from the paper
Fully-connected layers only (no convolutional BN). Test accuracy with BN fluctuates between epochs because inference uses running statistics.

## Run
```bash
python main.py
```

## Output
```text
max |difference| train-mode stats vs nn.BatchNorm1d: 9.54e-07
max |difference| eval-mode output                 : 9.54e-07

Test accuracy of a 6-layer sigmoid MLP (SGD, lr=0.2)
epoch   without BN   with BN
    1        0.096     0.407
    5        0.098     0.837
   10        0.100     0.726
   20        0.098     0.963
   30        0.106     0.885
```
