FROM continuumio/miniconda3:latest

# update & upgrade
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN conda install -y python=3.7.3 numpy scipy matplotlib cython \
 && conda clean -y --all

ARG CACHEBUST=1

# my packages
RUN pip install --no-cache-dir \
                git+https://github.com/JoshKarpel/simulacra.git \
                git+https://github.com/JoshKarpel/ionization.git \
 && python3 -c "import ionization"
