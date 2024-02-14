FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git git-lfs

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} duser
RUN useradd -l -u ${USER_ID} -g duser duser
RUN install -d -m 0755 -o duser -g duser /home/duser
RUN chown --changes --silent --no-dereference --recursive ${USER_ID}:${GROUP_ID} /home/duser

#Install python3 pip3
RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.10 python3.10-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN apt-get install -y python-apt --reinstall
RUN apt-get install -y htop iotop
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools wheel virtualenv

# Other system level tools
RUN apt-get install -y htop iotop

####### USER AREA ##########################################
USER duser
WORKDIR /home/duser

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888