#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python vrex.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/vrex/PACS_P
CUDA_VISIBLE_DEVICES=0 python vrex.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/vrex/PACS_A
CUDA_VISIBLE_DEVICES=0 python vrex.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/vrex/PACS_C
CUDA_VISIBLE_DEVICES=0 python vrex.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/vrex/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python vrex.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/vrex/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python vrex.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/vrex/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python vrex.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/vrex/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python vrex.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/vrex/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_i
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_q
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python vrex.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --trade-off 1 --seed 0 --log logs/vrex/DomainNet_s



#멀티소스
CUDA_VISIBLE_DEVICES=0 python vrex.py data/Cognex -d Cognex -s Custom_repeat Custom_cameraz Custom_light Custom_bright Custom_blur -t Repeat Brightness Cameraz Lcondition -a resnet50 --seed 0 --log logs/vrex/Cognex

CUDA_VISIBLE_DEVICES=0 python vrex.py data/Cognex -d Cognex -s Custom_repeat Custom_dropout -t Repeat Brightness Cameraz Lcondition -a resnet50 --log logs/vrex/Cognex --lr 0.01 --n-domains-per-batch 2





CUDA_VISIBLE_DEVICES=0 python vrex.py data/Cognex -d Cognex -s Custom_repeat Custom_bright Custom_cameraz -t Repeat Brightness Cameraz Lcondition -a resnet50 --log logs/vrex/Cognex --lr 0.01 --n-domains-per-batch 2