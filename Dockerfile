FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    build-essential \
    python3-pip \
    zip \
    unzip \
    tmux \
    tree \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --set python /usr/bin/python3.9

RUN python3.9 -m pip install --no-cache-dir --upgrade pip \
    && pip install ipykernel transformers datasets pillow sentencepiece protobuf deepspeed evaluate loguru wandb donut tqdm accelerate>=0.21.0 \
    && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 \
    && pip install zss nltk \
    && pip install transformers==4.28.1 pytorch-lightning \
    && pip install openai pyyaml argparse tqdm json5 matplotlib

WORKDIR /app

RUN mkdir -p /app/tests && chmod -R 755 /app

CMD ["python"]
