FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y  \
    python3-pip wget curl build-essential git wget vim zip \
    zlib1g-dev \
    unzip \
    pkg-config \
    libgl-dev \
    libblas-dev \
    liblapack-dev \
    python3-tk \
    python3-wheel \
    graphviz \
    libhdf5-dev \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    swig \
    apt-transport-https \
    lsb-release \
    libpng-dev \
    ca-certificates

RUN apt-get update && apt-get install nodejs -y

RUN apt-get clean && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.10 /usr/local/bin/python3 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* # keep the image clean

RUN python3 -m pip --no-cache-dir install \
    blackcellmagic\
    pytest \
    pytest-cov \
    numpy \
    matplotlib \
    scipy \
    pandas \
    jupyter \
    jupyterlab \
    scikit-learn \
    scikit-image \
    seaborn \
    graphviz \
    Pillow \
    opencv-python

RUN python3 -m pip --no-cache-dir install \
    torch torchvision torchaudio -f https://download.pytorch.org/whl/cu116/torch_stable.html

# install python requirements
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ cv_bridge rospy cv_bridge sensor_msgs std_msgs geometry_msgs

RUN pip3 install albumentations


ENV NODE_VERSION=16.13.0
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# Set up the ROS environment
ENV ROS_DISTRO melodic
ENV ROS_MASTER_URI http://127.0.0.1:11311
ENV ROS_HOSTNAME `hostname -i`

#EXPOSE 11311 8888

RUN jupyter labextension install @jupyterlab/debugger
#ENTRYPOINT ["jupyter", "lab", "--allow-root"]

# set up zsh
RUN apt-get update && apt-get install zsh nano -y

ENV zsh /usr/bin/zsh
ENV ZSH /root/.oh-my-zsh
ENV ZSH_CUSTOM /root/.oh-my-zsh/custom
ENV HOME /root

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | /usr/bin/zsh || true

RUN git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$ZSH_CUSTOM/themes/spaceship-prompt" --depth=1 && \
    ln -s "$ZSH_CUSTOM/themes/spaceship-prompt/spaceship.zsh-theme" "$ZSH_CUSTOM/themes/spaceship.zsh-theme"
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

RUN rm ~/.zshrc && \
    echo "export ROS_DISTRO=melodic\n"\
        "export ZSH=$HOME/.oh-my-zsh\n" \
        "ZSH_THEME=\"spaceship\"\n" \
        "plugins=(git zsh-autosuggestions zsh-syntax-highlighting)\n" \
        "source $ZSH/oh-my-zsh.sh\n" \
        "export ROS_HOSTNAME=`hostname -i`" | tee -a ~/.zshrc

ENTRYPOINT ["/usr/bin/zsh"]

# docker run --net=host --runtime=nvidia -v $(pwd):/robot -it roboticus
