# Learning Visual Representations of Colorectal Cancer in Histology Images using Contrastive Learning

## Summary
In this work, we compare the performance of several contrastive learning models, both supervised and unsupervised. The 3 models we experiment with are:
- Simple Contrastive Learning (SimCLR)
- Bootstrap Your Own Latents (BYOL)
- Supervised Contrastive Learning (SupCon)

We compare each model to a supervised benchmark model, the ResNet18 model. 

## How to reproduce results

### ResNet18 baseline
#### Train a pretrained model
Run `python3 run_models.py --model baseline --baseline resnet18 --mode train --pretrained True`

#### Train from scratch
Run `python3 run_models.py --model baseline --baseline resnet18 --mode train`

#### Test from epoch weights
Run `python3 run_models.py --model baseline --baseline resnet18 --mode test --from_epoch <EPOCH_PATH>`

### SimCLR
#### Train a pretrained model
Run `python3 run_models.py --model simclr --mode train --pretrained True`

#### Train from scratch
Run `python3 run_models.py --model simclr --mode train`

#### Test from epoch weights
Run `python3 run_models.py --model simclr --mode test --from_epoch <EPOCH_PATH>`

### BYOL
#### Train a pretrained model
Run `python3 run_models.py --model byol --mode train --pretrained True`

#### Train from scratch
Run `python3 run_models.py --model byol --mode train`

#### Test from epoch weights
Run `python3 run_models.py --model byol --mode test --from_epoch <EPOCH_PATH>`

### SupCon
#### Train a pretrained model
Run `python3 run_models.py --model supcon --mode train --pretrained True`

#### Train from scratch
Run `python3 run_models.py --model supcon --mode train`

#### Test from epoch weights
Run `python3 run_models.py --model supcon --mode test --from_epoch <EPOCH_PATH>`

## References
We consulted the following references while writing our code. 
- https://github.com/Spijkervet/SimCLR
- https://github.com/lucidrains/byol-pytorch
- https://github.com/HobbitLong/SupContrast