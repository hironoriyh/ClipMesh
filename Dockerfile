#Fpip ROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04 
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
USER root


ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.9
# ARG WITH_TORCHVISION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         ca-certificates \
	     apt-utils apt-transport-https pkg-config \
	     software-properties-common \
         libjpeg-dev \
    	 libopencv-dev \
         libpng-dev \
         wget git curl vim tmux \
         blender && \
	#  libopenexr-dev && \
	 # Blender and openSCAD are soft dependencies used in Trimesh for boolean operations with subprocess
     rm -rf /var/lib/apt/lists/*

# install conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch && \
     /opt/conda/bin/conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN find /opt/conda -type d -exec chmod 777 \{\} \;
RUN find /opt/conda -type f -exec chmod o+rw \{\} \;

# Install the base PIP packages.
RUN pip  install --upgrade pip 
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
RUN pip  install opencv-python setuptools numpy PyYAML Cython
RUN pip  install black==19.3.b0 flake8-comprehensions==3.3.0 flake8-bugbear==20.1.4
RUN pip  install flake8==3.8.4 isort==4.3.21 m2r2 mccabe==0.6.1 mock sphinx
RUN pip  install sphinx_markdown_tables sphinx_rtd_theme argparse tqdm tensorboard
RUN pip  install open3d trimesh pymcubes matplotlib scipy
RUN pip  install ipdb trimesh pymeshlab
RUN pip  install ftfy regex tqdm
RUN pip  install git+https://github.com/openai/CLIP.git

# RUN pip  install "git+https://github.com/facebookresearch/pytorch3d.git"

## you need to install manifold software: https://github.com/ranahanocka/Point2Mesh/#install-manifold-software
# RUN cd /tmp && \
#     git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git && \
#     cd Manifold && \
#     mkdir build && \
#     cd build && \
#     cmake .. -DCMAKE_BUILD_TYPE=Release && \
#     make

# COPY ./loop_limitation /tmp/loop_limitation
# RUN cd /tmp/loop_limitation && \
#     pip install . &&\
#     rm -r /tmp/loop_limitation

# COPY ./DALLE2-pytorch /tmp/DALLE2-pytorch
# RUN cd /tmp/DALLE2-pytorch && \
#     pip install . &&\
#     rm -r /tmp/DALLE2-pytorch

# RUN mkdir /tmp/weights
# RUN wget https://huggingface.co/spaces/NasirKhalid24/Dalle2-Diffusion-Prior/resolve/main/larger-model.pth -O /tmp/weights/model.pth


RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev mesa-utils && \
    rm -rf /var/lib/apt/lists/*

#RUN groupadd -g 1001 hyoshida && \
#    useradd -m -s /bin/bash -u 1001 -g 1001 hyoshida
#USER hyoshida
#WORKDIR /home/$USERNAME/
#RUN echo "source activate base" >> /home/hyoshida/.bashrc

RUN groupadd -g 1002 hozumi && \
    useradd -m -s /bin/bash -u 1002 -g 1002 hozumi
USER hozumi
WORKDIR /home/$USERNAME/

RUN echo "source activate base" >> ~/.bashrc

#RUN ln -s /tmp/weights/model.pth 
