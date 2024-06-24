#FROM ubuntu:20.04
#FROM nvidia/cuda:11.8.0-base-ubuntu20.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1

RUN apt update && apt upgrade -y \
  && apt install -y \
        python3-pip \
        libgl1-mesa-dev \
        git \
        git-lfs \
        wget \
        curl \
        # sudo \
        # graphviz \
        fonts-ipaexfont \
        vim \
        ffmpeg \
        build-essential \
        cmake \
        zip \
        unzip \
        jq \
  && pip install -U pip \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/own/build
RUN chmod -R 777 /tmp/own
WORKDIR /tmp/own/build

# ユーザーを作成
ARG DOCKER_UID=1000
ARG DOCKER_USER=docker
ARG DOCKER_PASSWORD=docker
RUN useradd -m \
  --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

USER ${DOCKER_USER}

# Miniconda setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /tmp/own/conda \
  && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /tmp/own/conda/bin:$PATH

RUN conda create -n Dev python=3.10
RUN conda init bash
RUN echo "conda activate Dev" >> ~/.bashrc

ENV PATH /tmp/own/conda/envs/Dev/bin:$PATH

# Activate the conda environment and install the Python dependencies
#RUN /bin/bash -c "source ~/.bashrc && pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118"
RUN /bin/bash -c "source ~/.bashrc && pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121"

COPY ./requirements-basic.txt /tmp/own/build/
RUN /bin/bash -c "source ~/.bashrc && pip install --no-cache-dir -r requirements-basic.txt"

RUN pip install einops

RUN pip install deepspeed

RUN pip install peft bitsandbytes

RUN pip install wandb

RUN pip install gradio

RUN pip install PyMuPDF

RUN mkdir -p /tmp/own/docker/entrypoint
COPY docker-entrypoint.sh /tmp/own/docker/entrypoint

ENTRYPOINT ["bash", "/tmp/own/docker/entrypoint/docker-entrypoint.sh"]

CMD [ "/bin/bash" ]
