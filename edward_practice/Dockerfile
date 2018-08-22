FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y libmecab-dev \
    && apt-get install -y git\
    && apt-get install -y make\
    && apt-get install -y curl\
    && apt-get install -y xz-utils\
    && apt-get install -y file\
    && apt-get install -y sudo\
    && apt-get install -y wget

RUN apt-get update \
	&& apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update


RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN python3.6 -m pip install pip --upgrade


COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt


ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en


WORKDIR /work


CMD ['/bin/bash']
