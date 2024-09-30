FROM --platform=linux/amd64 python:3.12
RUN python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install wandb==0.18.2 omegaconf==2.3.0 hydra-core==1.3.2 
