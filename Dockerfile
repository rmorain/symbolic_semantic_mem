FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

FROM fastai/codespaces

FROM pytorch/pytorch

WORKDIR /kirby

ADD . /kirby

ENV PACKAGES="\
    git \
"

RUN apt update && apt -y install ${PACKAGES}

RUN pip install -r requirements.txt

ENV NAME kirby 
