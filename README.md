# CS6101
Try colossal AI framework to train a model.

## Environment requirement
+ torch
+ torchvision
+ colossalai
+ timm

## model config
ViT model

depth=12，num_heads = 6

trained with 2 gpu on cifar-10

run  `colossalai run --nproc_per_node 2 train_vit.py --config config_vit.py`

Accuracy: 96.2%
