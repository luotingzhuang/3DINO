FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ARG USER_ID
ARG GROUP_ID

ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# xFormers is required by train/ssl_meta_arch.py. This version matches torch 2.4.1.
RUN python -m pip install xformers==0.0.28.post1 --no-deps

RUN python -m pip install \
    fvcore \
    monai==1.4.0 \
    nibabel \
    omegaconf \
    scikit-learn \
    torchio \
    torchmetrics

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} luotingzhuang && \
    useradd -l -u ${USER_ID} -g luotingzhuang luotingzhuang && \
    install -d -m 0755 -o luotingzhuang -g luotingzhuang /home/luotingzhuang;fi


USER luotingzhuang