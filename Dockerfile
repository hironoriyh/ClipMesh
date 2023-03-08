FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER root
WORKDIR /tmp

ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         ca-certificates \
	 apt-utils apt-transport-https pkg-config \
	 software-properties-common \
         libjpeg-dev \
	 libopencv-dev \
         libpng-dev \
	 libopenexr-dev \
	 # Blender and openSCAD are soft dependencies used in Trimesh for boolean operations with subprocess
	 vim tmux git wget curl && \
     rm -rf /var/lib/apt/lists/*


#RUN find /opt/conda -type d -exec chmod 777 \{\} \;
#RUN find /opt/conda -type f -exec chmod o+rw \{\} \;


# Install the base PIP packages.
SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install --upgrade pip

RUN pip install opencv-python setuptools numpy PyYAML Cython
RUN pip install black==19.3.b0 flake8-comprehensions==3.3.0 flake8-bugbear==20.1.4
RUN pip install flake8==3.8.4 isort==4.3.21 m2r2 mccabe==0.6.1 mock sphinx
RUN pip install sphinx_markdown_tables sphinx_rtd_theme argparse tqdm tensorboard
# RUN pip install open3d trimesh pymcubes
RUN pip install optuna
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install the remaining dependencies.
RUN pip install ipdb

# install meshlabserver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    xvfb libglu1-mesa libglib2.0-0 libgomp1 \
        libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /tmp/requirements.txt


RUN cd /tmp && \
   #/bin/bash -c "source activate base" && \
    pip install -r requirements.txt

COPY  loop_limitation/ /tmp/loop_limitation/ 
RUN   cd /tmp/loop_limitation && \
    # /bin/bash -c "source activate base" && \
      pip install . --user 

COPY  DALLE2-pytorch/ /tmp/DALLE2-pytorch/ 
RUN   cd /tmp/DALLE2-pytorch && \
     #/bin/bash -c "source activate base" && \
      pip install . --user 



### nvdiffrast

COPY ./nvdiffrast/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

#RUN /bin/bash -c "source activate base" && \
RUN pip install ninja imageio imageio-ffmpeg

COPY ./nvdiffrast /tmp/pip/nvdiffrast/
COPY ./nvdiffrast/setup.py /tmp/pip/

RUN cd /tmp/pip/nvdiffrast && \
#    /bin/bash -c "source activate base" && \
    pip install .

ENV CUDA_VERSION=11.3
ENV CUDA_HOME=/usr/local/cuda-11.3

